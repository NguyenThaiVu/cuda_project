# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="int8mm_ext",
    ext_modules=[
        CUDAExtension(
            name="int8mm_ext",
            sources=["int8mm.cpp", "int8mm_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-DBLOCK_SIZE=32"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
