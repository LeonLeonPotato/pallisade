import torch.multiprocessing as mp
import torch

def foo(worker,tl):
    print(tl)
    tl[worker] += (worker+1) * 1000

if __name__ == '__main__':
    mp.set_start_method("spawn")
    tl = [torch.randn(2).to(device="cuda"), torch.randn(3).to(device="cuda")]

    for t in tl:
        t.share_memory_()

    print("before mp: tl=")
    print(tl)

    p0 = mp.Process(target=foo, args=(0, tl))
    p1 = mp.Process(target=foo, args=(1, tl))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    print("after mp: tl=")
    print(tl)