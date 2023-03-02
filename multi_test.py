import torch
import torch.multiprocessing as mp

def work(queue):
    tmp = queue.get()
    obj = tmp.clone()
    del tmp
    print(obj)

if __name__ == '__main__':
    mp.set_start_method("spawn")
    queue = mp.SimpleQueue()
    
    test = torch.tensor([69, 420])

    queue.put(test.share_memory_())
    proc = mp.Process(
        target=work, args=(queue, )
    )
    proc.start()
    proc.join()
    print(test)