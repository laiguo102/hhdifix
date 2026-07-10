import pyiqa
import os
import argparse
from pathlib import Path
import torch
import util_image
import tqdm
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import json
import pandas as pd

print(pyiqa.list_models())
def evaluate(in_path, ref_path, ntest):
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_dict["musiq_aesthetic"] = pyiqa.create_metric('musiq-ava').to(device)
    metric_dict["musiq_spaq"] = pyiqa.create_metric('musiq-spaq').to(device)
    metric_dict["musiq_paq2piq"] = pyiqa.create_metric('musiq-paq2piq').to(device)
    metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device)
    metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device)
    metric_dict["maniqa_kadid"] = pyiqa.create_metric('maniqa-kadid').to(device)
    metric_paired_dict = {}
    
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()
    
    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None: ref_path_list = ref_path_list[:ntest]
        
        metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
        metric_paired_dict["dists"]=pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)
        
    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None: lr_path_list = lr_path_list[:ntest]
    
    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = {}
    resultall = {
    'in_name': [],
    'ssim': [],
    'psnr': [],
    'lpips': [],
    'musiq': [],
    'musiq_aesthetic': [],
    'musiq_spaq': [],
    'musiq_paq2piq': [],
    'maniqa': [],
    'maniqa_kadid': [],
    'clipiqa': [],
    'dists': [],
    'niqe': []
}
    for i in tqdm.tqdm(range(len(lr_path_list))):
        _in_path = lr_path_list[i]
        _ref_path = ref_path_list[i] if ref_path_list is not None else None
        
        resultall['in_name'].append(os.path.basename(_in_path))
        im_in = util_image.imread(_in_path, chn='rgb', dtype='float32')  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()              # 1 x c x h x w
        im_in_tensor = F.interpolate(im_in_tensor, size=(512, 512), mode='bilinear', align_corners=False)
        #im_in_tensor = F.interpolate(im_in_tensor, size=(384, 384), mode='bilinear', align_corners=False)
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                res = metric(im_in_tensor).item()
                result[key] = result.get(key, 0) + res
                resultall[key].append(res)
        
        if ref_path is not None:
            im_ref = util_image.imread(_ref_path, chn='rgb', dtype='float32')  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()    
            im_ref_tensor = F.interpolate(im_ref_tensor, size=(512, 512), mode='bilinear', align_corners=False)
            #im_ref_tensor = F.interpolate(im_ref_tensor, size=(384, 384), mode='bilinear', align_corners=False)
            for key, metric in metric_paired_dict.items():
                res = metric(im_in_tensor, im_ref_tensor).item()
                result[key] = result.get(key, 0) + res
                resultall[key].append(res)
    
    if ref_path is not None:
        fid_metric = pyiqa.create_metric('fid')
        result['fid'] = fid_metric(in_path, ref_path)

    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
        else:
            print(f"{key}: {res/len(lr_path_list):.5f}")
    
    # 在打印完平均值之后，保存 CSV
    # 将 resultall 字典转换为 DataFrame
    df = pd.DataFrame(resultall)
    
    # 重新排列一下列的顺序，把 'in_name' 放在第一列
    cols = ['in_name'] + [c for c in df.columns if c != 'in_name']
    df = df[cols]
    
    # 保存到当前目录
    csv_path = "image_metrics_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {os.path.abspath(csv_path)}")

    jsonpath = os.path.join(args.output)
    with open(jsonpath, 'w') as f:
        json.dump(resultall, f, indent=4)  # indent=4 让文件可读性更好
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--in_path", type=str, required=True)
    parser.add_argument("-r", "--ref_path", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default='./')
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.in_path, args.ref_path, args.ntest)
    
