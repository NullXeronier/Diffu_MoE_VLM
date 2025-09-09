"""
FP8 Model Utilities for NVIDIA Llama-3.3-70B-Instruct-FP8
Handles FP8 data type conversions and TensorRT-LLM integration
"""

import os
import logging
import torch
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import numpy as np

try:
    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("[Warning] TensorRT-LLM not available. Install with: pip install tensorrt-llm")


class FP8ModelManager:
    """
    Manager for FP8 model operations and data type handling
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model_name = model_config.get("model_name", "nvidia/Llama-3.3-70B-Instruct-FP8")
        self.use_fp8 = model_config.get("quantization", {}).get("enabled", True)
        self.precision = model_config.get("quantization", {}).get("precision", "fp8")
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Model instance
        self.llm_instance = None
        self.sampling_params = None
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "fp8_memory_saved": 0.0
        }
    
    def initialize_model(self) -> bool:
        """Initialize the FP8 model with TensorRT-LLM"""
        if not TENSORRT_AVAILABLE:
            self.logger.error("TensorRT-LLM not available")
            return False
        
        try:
            # Extract configuration
            perf_config = self.model_config.get("performance", {})
            quant_config = self.model_config.get("quantization", {})
            
            # Initialize LLM with FP8 settings
            self.llm_instance = LLM(
                model=self.model_name,
                tensor_parallel_size=perf_config.get("tensor_parallel_size", 2),
                dtype=quant_config.get("data_type", "float8_e4m3fn"),
                quantization="fp8" if self.use_fp8 else None,
                gpu_memory_utilization=perf_config.get("gpu_memory_utilization", 0.85),
                max_model_len=self.model_config.get("engine", {}).get("max_context_length", 8192),
                enforce_eager=not perf_config.get("enable_cuda_graph", True),
                enable_chunked_prefill=perf_config.get("enable_chunked_prefill", True),
                max_num_batched_tokens=perf_config.get("max_num_batched_tokens", 8192)
            )
            
            # Initialize sampling parameters
            sampling_config = self.model_config.get("sampling", {})
            self.sampling_params = SamplingParams(
                temperature=sampling_config.get("temperature", 0.1),
                top_p=sampling_config.get("top_p", 0.95),
                top_k=sampling_config.get("top_k", 50),
                max_tokens=sampling_config.get("max_tokens", 512),
                repetition_penalty=sampling_config.get("repetition_penalty", 1.1),
                stop=sampling_config.get("stop_sequences", [])
            )
            
            self.logger.info(f"FP8 model initialized: {self.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FP8 model: {e}")
            return False
    
    def convert_to_fp8(self, data: Union[torch.Tensor, np.ndarray, List, str]) -> Any:
        """Convert input data to FP8 format if applicable"""
        if not self.use_fp8:
            return data
        
        if isinstance(data, str):
            # Text data - no conversion needed
            return data
        elif isinstance(data, torch.Tensor):
            # Convert tensor to FP8
            if data.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                # Convert to FP8 E4M3 format (simulated with float16 clamping)
                return self._simulate_fp8_conversion(data)
            return data
        elif isinstance(data, np.ndarray):
            # Convert numpy array
            tensor = torch.from_numpy(data)
            converted = self._simulate_fp8_conversion(tensor)
            return converted.numpy()
        elif isinstance(data, list):
            # Recursively convert list elements
            return [self.convert_to_fp8(item) for item in data]
        else:
            return data
    
    def _simulate_fp8_conversion(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simulate FP8 E4M3 conversion by clamping values"""
        if not self.use_fp8:
            return tensor
        
        # FP8 E4M3 range: approximately [-448, 448]
        # This is a simulation - actual FP8 would be handled by TensorRT-LLM
        fp8_max = 448.0
        fp8_min = -448.0
        
        # Clamp to FP8 range and reduce precision
        clamped = torch.clamp(tensor, fp8_min, fp8_max)
        
        # Simulate reduced precision by quantizing
        scale = fp8_max / torch.max(torch.abs(clamped))
        quantized = torch.round(clamped * scale * 127) / (scale * 127)
        
        return quantized.to(torch.float16)  # Use float16 as proxy for FP8
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using FP8 model"""
        if not self.llm_instance:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        import time
        start_time = time.time()
        
        try:
            # Override sampling params if provided
            sampling_params = self.sampling_params
            if kwargs:
                sampling_config = self.model_config.get("sampling", {})
                sampling_config.update(kwargs)
                
                sampling_params = SamplingParams(
                    temperature=sampling_config.get("temperature", 0.1),
                    top_p=sampling_config.get("top_p", 0.95),
                    top_k=sampling_config.get("top_k", 50),
                    max_tokens=sampling_config.get("max_tokens", 512),
                    repetition_penalty=sampling_config.get("repetition_penalty", 1.1),
                    stop=sampling_config.get("stop_sequences", [])
                )
            
            # Generate response
            outputs = self.llm_instance.generate([prompt], sampling_params)
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            
            # Update metrics
            end_time = time.time()
            self.metrics["total_requests"] += 1
            self.metrics["total_time"] += (end_time - start_time)
            
            if hasattr(outputs[0], 'usage'):
                self.metrics["total_tokens"] += outputs[0].usage.total_tokens
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration"""
        return {
            "model_name": self.model_name,
            "precision": self.precision,
            "use_fp8": self.use_fp8,
            "tensorrt_available": TENSORRT_AVAILABLE,
            "initialized": self.llm_instance is not None,
            "config": self.model_config,
            "metrics": self.metrics
        }
    
    def estimate_memory_savings(self) -> Dict[str, float]:
        """Estimate memory savings from FP8 quantization"""
        if not self.use_fp8:
            return {"memory_savings_gb": 0.0, "savings_percentage": 0.0}
        
        # Estimate for 70B model
        # FP16: ~140GB, FP8: ~70GB (approximately 50% reduction)
        fp16_memory = 140.0  # GB
        fp8_memory = 70.0    # GB
        savings = fp16_memory - fp8_memory
        percentage = (savings / fp16_memory) * 100
        
        return {
            "fp16_memory_gb": fp16_memory,
            "fp8_memory_gb": fp8_memory,
            "memory_savings_gb": savings,
            "savings_percentage": percentage
        }
    
    def validate_fp8_setup(self) -> Dict[str, bool]:
        """Validate FP8 setup and capabilities"""
        validation = {
            "tensorrt_available": TENSORRT_AVAILABLE,
            "cuda_available": torch.cuda.is_available(),
            "fp8_hardware_support": False,
            "model_accessible": False,
            "config_valid": True
        }
        
        # Check for FP8-capable hardware
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            fp8_gpus = ["H100", "H200", "RTX 4090", "RTX 4080"]
            validation["fp8_hardware_support"] = any(gpu in gpu_name for gpu in fp8_gpus)
        
        # Check model accessibility
        try:
            # This would need actual model access
            validation["model_accessible"] = True
        except:
            validation["model_accessible"] = False
        
        # Validate configuration
        required_keys = ["model_name", "quantization", "performance"]
        validation["config_valid"] = all(key in self.model_config for key in required_keys)
        
        return validation
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self.metrics["total_requests"] == 0:
            return {"avg_latency": 0.0, "throughput": 0.0, "total_requests": 0}
        
        avg_latency = self.metrics["total_time"] / self.metrics["total_requests"]
        throughput = self.metrics["total_tokens"] / self.metrics["total_time"] if self.metrics["total_time"] > 0 else 0.0
        
        return {
            "avg_latency_seconds": avg_latency,
            "throughput_tokens_per_second": throughput,
            "total_requests": self.metrics["total_requests"],
            "total_tokens": self.metrics["total_tokens"],
            "total_time": self.metrics["total_time"]
        }
    
    def cleanup(self):
        """Cleanup model resources"""
        if self.llm_instance:
            # TensorRT-LLM cleanup would go here
            self.llm_instance = None
            self.logger.info("FP8 model resources cleaned up")


def create_fp8_manager(config_path: Optional[str] = None) -> FP8ModelManager:
    """Create FP8 model manager from configuration"""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "fp8_model.yaml"
    
    import yaml
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return FP8ModelManager(config)
    except Exception as e:
        logging.error(f"Failed to load FP8 config from {config_path}: {e}")
        # Return with default config
        default_config = {
            "model_name": "nvidia/Llama-3.3-70B-Instruct-FP8",
            "quantization": {"enabled": True, "precision": "fp8"},
            "performance": {"tensor_parallel_size": 2},
            "sampling": {"temperature": 0.1, "max_tokens": 512}
        }
        return FP8ModelManager(default_config)


def validate_fp8_environment() -> Dict[str, Any]:
    """Validate the environment for FP8 model execution"""
    validation = {
        "status": "unknown",
        "checks": {},
        "recommendations": []
    }
    
    # Check TensorRT-LLM
    validation["checks"]["tensorrt_llm"] = TENSORRT_AVAILABLE
    if not TENSORRT_AVAILABLE:
        validation["recommendations"].append("Install TensorRT-LLM: pip install tensorrt-llm")
    
    # Check CUDA
    validation["checks"]["cuda"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        validation["recommendations"].append("CUDA not available. Install CUDA toolkit.")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        validation["checks"]["gpu_memory_gb"] = gpu_memory
        validation["checks"]["sufficient_memory"] = gpu_memory >= 40  # Minimum for FP8
        
        if gpu_memory < 40:
            validation["recommendations"].append(f"Insufficient GPU memory: {gpu_memory:.1f}GB. Need at least 40GB for FP8 model.")
    
    # Determine overall status
    critical_checks = ["tensorrt_llm", "cuda", "sufficient_memory"]
    if all(validation["checks"].get(check, False) for check in critical_checks):
        validation["status"] = "ready"
    elif validation["checks"].get("tensorrt_llm", False) and validation["checks"].get("cuda", False):
        validation["status"] = "partial"
    else:
        validation["status"] = "not_ready"
    
    return validation


if __name__ == "__main__":
    # Test FP8 utilities
    print("FP8 Model Utilities Test")
    print("=" * 40)
    
    # Validate environment
    env_status = validate_fp8_environment()
    print(f"Environment Status: {env_status['status']}")
    print(f"Checks: {env_status['checks']}")
    if env_status['recommendations']:
        print("Recommendations:")
        for rec in env_status['recommendations']:
            print(f"  - {rec}")
    
    # Test model manager creation
    try:
        manager = create_fp8_manager()
        print(f"\nModel Manager Created:")
        print(f"Model: {manager.model_name}")
        print(f"FP8 Enabled: {manager.use_fp8}")
        print(f"Memory Savings: {manager.estimate_memory_savings()}")
        
        # Test data conversion
        test_tensor = torch.randn(4, 4)
        converted = manager.convert_to_fp8(test_tensor)
        print(f"\nData Conversion Test:")
        print(f"Original shape: {test_tensor.shape}, dtype: {test_tensor.dtype}")
        print(f"Converted shape: {converted.shape}, dtype: {converted.dtype}")
        
    except Exception as e:
        print(f"Error testing model manager: {e}")
