import unittest
import torch
from torchvision.ops import MultiScaleRoIAlign
from deterministic_gpu import DeterministicRoIAlign


class DeterministicPoolTests(unittest.TestCase):
    def test_reference_matches_native_cpu_geometry_and_gradient(self):
        torch.manual_seed(12)
        x=torch.rand(2,3,24,32,requires_grad=True);y=x.detach().clone().requires_grad_()
        boxes=[torch.tensor([[0.,0,128,96],[13.,7,65,43]]),torch.tensor([[25.,10,110,80]])]
        native=MultiScaleRoIAlign(['0'],7,2);reference=DeterministicRoIAlign(['0'],7,2)
        a=native({'0':x},boxes,[(96,128)]*2);b=reference({'0':y},boxes,[(96,128)]*2)
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=1e-6));a.sum().backward();b.sum().backward()
        self.assertTrue(torch.allclose(x.grad,y.grad,atol=1e-5,rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_reference_gpu_backward_exact_repeat(self):
        old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True,warn_only=False);torch.manual_seed(13)
            base=torch.rand(2,3,24,32,device='cuda')
            boxes=[torch.tensor([[0.,0,128,96],[13.,7,65,43]],device='cuda'),torch.tensor([[25.,10,110,80]],device='cuda')]
            outputs=[];gradients=[]
            for _ in range(2):
                x=base.clone().requires_grad_();pool=DeterministicRoIAlign(['0'],7,2)
                value=pool({'0':x},boxes,[(96,128)]*2);value.square().sum().backward();outputs.append(value);gradients.append(x.grad)
            self.assertTrue(torch.equal(outputs[0],outputs[1]));self.assertTrue(torch.equal(gradients[0],gradients[1]))
        finally:torch.use_deterministic_algorithms(old,warn_only=warn)


if __name__=='__main__':unittest.main()
