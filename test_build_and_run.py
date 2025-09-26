import torch, int8mm_ext
A = torch.randint(-128,127,(512,512),dtype=torch.int8,device='cuda')
B = torch.randint(-128,127,(512,512),dtype=torch.int8,device='cuda')
C = int8mm_ext.int8_gemm(A,B)
print(C.shape, C.dtype)  # torch.Size([512, 512]) torch.int32
