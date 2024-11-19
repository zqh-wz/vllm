from typing import Any, Dict, List, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import PackedvLLMParameter, RowvLLMParameter, GroupQuantScaleParameter

import elevator
import json
import os

logger = init_logger(__name__)

class ElevatorConfig(QuantizationConfig):
    """Config class for Elevator"""

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
    ):
        self.weight_bits = weight_bits
        self.pack_factor = 32 // weight_bits
        self.group_size = group_size

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ElevatorConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if user_quant == "elevator":
            logger.warning("Using elevator kernel.")
            return user_quant
        return None
    
    @classmethod
    def get_min_capability(cls) -> int:
        return 70
    
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16]
    
    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["ElevatorLinearMethod"]:
        if isinstance(layer, LinearBase):
            return ElevatorLinearMethod(self, prefix)
        return None
    
    def get_name(self) -> str:
        return "elevator"

    @staticmethod
    def get_config_filenames() -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError
    

class ElevatorLinearMethod(LinearMethodBase):
    """Linear method for Elevator.

    Args:
        quant_config: The Elevator quantization config.
    """

    def __init__(self, quant_config: ElevatorConfig, prefix: str):
        self.quant_config = quant_config
        self.prefix = prefix
        self.kernel_config = json.load(open("best_config.json", "r"))
        for key in self.kernel_config:
            self.kernel_config[key]["args"] = json.loads(self.kernel_config[key]["args"])
    
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader
        )

        g_idx = RowvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader
        )

        scales_and_zp_size = input_size_per_partition // self.quant_config.group_size

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader
        )
        qzeros = PackedvLLMParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)
    
    def process_weights_after_loading(self, layer: torch.nn.Module):
        print(self.prefix)
        input_size = layer.qweight.shape[0] * self.quant_config.pack_factor
        output_size = layer.qweight.shape[1]
        device = layer.qweight.device

        if os.path.exists(f".elevator_linears/{self.prefix}.pt"):
            qweight_uncompressed, qzeros_uncompressed = torch.load(f".elevator_linears/{self.prefix}.pt", weights_only=True)
        else:
            row_idx = torch.arange(input_size, device=device, dtype=torch.int32) // self.quant_config.pack_factor
            row_offset = torch.arange(input_size, device=device, dtype=torch.int32) % self.quant_config.pack_factor
            mask = (1 << self.quant_config.weight_bits) - 1
            qweight_uncompressed = (layer.qweight[row_idx] >> (row_offset * self.quant_config.weight_bits).unsqueeze(1)) & mask

            col_idx = torch.arange(output_size, device=device, dtype=torch.int32) // self.quant_config.pack_factor
            col_offset = torch.arange(output_size, device=device, dtype=torch.int32) % self.quant_config.pack_factor
            qzeros_uncompressed = (layer.qzeros[:, col_idx] >> (col_offset * self.quant_config.weight_bits).unsqueeze(0)) & mask

            os.makedirs(".elevator_linears", exist_ok=True)
            torch.save((qweight_uncompressed, qzeros_uncompressed), f".elevator_linears/{self.prefix}.pt")

        config = self.kernel_config[f"1-{output_size}-{input_size}-{self.quant_config.group_size}"]

        layer.elevator_linear = elevator.auto_get_layer(
            kernel_name=config["kernel"],
            template_args=config["args"],
            init_params=(
                qweight_uncompressed,
                layer.scales,
                qzeros_uncompressed,
            ),
            device=device,
        )

        layer.register_parameter("qweight", None)
        layer.register_parameter("qzeros", None)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        return layer.elevator_linear(x)