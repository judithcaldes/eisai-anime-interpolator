from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch
import _util.distance_transform_v0 as udist

import argparse  

device = torch.device('cuda')

# Cargar imágenes
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  
    return F.to_tensor(image).unsqueeze(0).to(device)  

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pred', type=str)
    ap.add_argument('gt', type=str)
    args = ap.parse_args()

    # Cargar imágenes
    pred = load_image(args.pred)
    gt = load_image(args.gt)

    # Load metrics
    metrics = torchmetrics.MetricCollection({
        'psnr': utorch.PSNRMetricCPU(),
        'ssim': utorch.SSIMMetricCPU(),
        'lpips': utorch.LPIPSMetric(net_type='alex'),
        'chamfer': udist.ChamferDistance2dMetric(t=2.0, sigma=1.0),
        # 'chamfer': udist.ChamferDistance2dMetric(block=512, t=2.0, sigma=1.0),
    }).to(device).eval()

    # Evaluar
    results = metrics(pred, gt)

    # Imprimir resultados
    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v.item()}")