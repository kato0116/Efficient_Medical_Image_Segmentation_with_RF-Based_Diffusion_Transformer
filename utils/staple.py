import torch

def mv(a):
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    return torch.sum(a, 0, keepdim=True) / b

def staple(a):
    # a: n,c,h,w detach tensor
    mvres = mv(a)
    gap = 0.4
    if gap > 0.02:
        for i, s in enumerate(a):
            r = s * mvres
            res = r if i == 0 else torch.cat((res,r),0)
        nres = mv(res)
        gap = torch.mean(torch.abs(mvres - nres))
        mvres = nres
        a = res
    return mvres