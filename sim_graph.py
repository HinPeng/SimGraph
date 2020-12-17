from __future__ import print_function

import os
import logger
import logging

from graph import Graph

class Config(object):
  __slots__ = (
    'device',
    'out_dir',
    'figure_dir',
    'mem_f',
    'net_name',
    'bs',
    'step_id',
    'run_prefix',
    'uname',
    'alloc_f',
    'temp_mem_f',
    'pers_mem_f',
    'new_net',
  )

  def __init__(self, **kwargs):
    super(Config, self).__init__()
    self.device = 'p100'
    self.out_dir = './graph'
    self.figure_dir = './figures'
    self.mem_f = './graph/mem.json'

    self.net_name = None
    self.bs = -1
    self.step_id = -1
    self.run_prefix = ''
    self.uname = None  # a unique name '{run_prefix}_{net_name}_{bs}'

    # dump file name
    # FileName: <train_or_eval>_netname_batchsize_stepid_alloc.log: (allocation_time, allocation_bytes)
    self.alloc_f = ''

    # these two files only write once
    # FileName: <train_or_eval>_netname_batchsize_tmpmem.log: (memory_size)
    self.temp_mem_f = ''
    # FileName: <train_or_eval>_netname_batchsize_persmem.log: (memory_size)
    self.pers_mem_f = ''

    # if running a new network: treat multiple iterations of the same networks as the same net
    self.new_net = True

    for f, v in kwargs.items():
      setattr(self, f, v)
    
    self.out_dir += '_{}'.format(self.device)
    self.figure_dir += '_{}'.format(self.device)

    if self.net_name:
      self.uname = '{}_{}_{}'.format(self.run_prefix, self.net_name, self.bs)

      self.out_dir = self.out_dir + '/{}'.format(self.net_name)
      self.figure_dir = self.figure_dir + '/{}'.format(self.net_name)

      if not os.path.exists(self.out_dir):
        os.makedirs(self.out_dir, exist_ok=True)
      if not os.path.exists(self.figure_dir):
        os.makedirs(self.figure_dir, exist_ok=True)

      self.alloc_f = '{}/{}_{}_{}_{}_alloc.log'.format(self.out_dir,
                                                       self.run_prefix,
                                                       self.net_name, self.bs,
                                                       self.step_id)
      self.temp_mem_f = '{}/{}_{}_{}_tmpmem.log'.format(
          self.out_dir, self.run_prefix, self.net_name, self.bs)
      self.pers_mem_f = '{}/{}_{}_{}_persmem.log'.format(
          self.out_dir, self.run_prefix, self.net_name, self.bs)

      if os.path.exists(self.temp_mem_f) and os.path.exists(self.pers_mem_f):
        self.new_net = False
    else:
      logging.error('name of networks should not be None')

  def issame(self, netname):
    return netname == self.uname

class SimGraph(object):

  def __init__(self, cfg):
    self._cfg = cfg
    self.graph = Graph(self._cfg)
    self.graph.InitFromFile()
    self.graph.CalculatePeakMem()
    self.graph.AccurateMemUsage()
    # self.graph.GetMemUsage()
    

  # no use
  # def InitTensorAlloc(self):
  #   class AllocInfo(object):
  #     def __init__(self, tensor_name, alloc_time, alloc_bytes):
  #       self.tensor_name = tensor_name
  #       self.time = alloc_time
  #       self.bytes = alloc_bytes

  #     def __eq__(self, other):
  #       return self.time == other.time and self.bytes == other.bytes and self.tensor_name == other.tensor_name

  #     def __gt__(self, other):
  #       if self.time > other.time:
  #         return True
  #       elif self.time == other.time:
  #         if self.bytes < other.bytes:
  #           return True
  #         elif self.bytes == other.bytes:
  #           return self.tensor_name > other.tensor_name
  #         else:
  #           return False
  #       else:
  #         return False

  #     def __str__(self):
  #       return '{}\t{}\t{}'.format(self.tensor_name, self.time, self.bytes)
  #   # init each tensor's deallocation postion (which node)
  #   for t in self.graph.tensors.values():
  #     if t.is_pers:
  #       continue
  #     if not t.is_alloc:
  #       assert t.shared_tensor != None
  #       continue

  #     node = self.graph.GetNode(t.node_name)
  #     t.alloc_time = node.start_time
  #     dealloc_time = node.end_time
      
        
  #     for i, t_out_node in enumerate(t.out_nodes):
  #       if t_out_node.end_time > dealloc_time:
  #         dealloc_time = t_out_node.end_time
          
  #     t.dealloc_time = dealloc_time
  #     # if dealloc_index != -1:
  #     #   t.out_nodes[dealloc_index].deallocs[t.name] = t
  #     # else:
  #     #   node.deallocs[t.name] = t
  #     self.allocations.append(AllocInfo(t.name, t.alloc_time, t.size))
  #     self.allocations.append(AllocInfo(t.name, t.dealloc_time, -t.size))

  #   # ordered_tensors = sorted(list(self.graph.tensors.values()), key=lambda x: x.name)
  #   # with open('./tensor_outnodes.log', 'w') as fout:
  #   #   for t in ordered_tensors:
  #   #     fout.write('{}:{}\n'.format(t.name, len(t.out_nodes)))

  #   for node in self.nodes:
  #     for i, t in enumerate(node.temp_allocs):
  #       temp_name = node.name+':t'+str(i)
  #       self.allocations.append(AllocInfo(temp_name, node.start_time, t.size))
  #       self.allocations.append(AllocInfo(temp_name, node.end_time, -t.size))

  #   self.allocations.sort()

  #   # debug
  #   with open('./alloc.log', 'w') as fout:
  #     for alloc in self.allocations:
  #       fout.write('{}\n'.format(str(alloc)))

  # def __call__(self):
  #   peak_mem = 0
  #   curr_mem = 0

  #   # self.allocations = []
  #   # with open('./alloc-node22.log') as fin:
  #   #   for line in fin:
  #   #     tmp = line.split('\t')
  #   #     self.allocations.append((int(tmp[0]), float(tmp[1])))

  #   for alloc in self.allocations:
  #     curr_mem += float(alloc.bytes) / (1<<10)
  #     if curr_mem > peak_mem:
  #       peak_mem = curr_mem

  #   logging.info('Peak memory is {} MB'.format(peak_mem))

def main():
  models = {
    'resnet50': 64,
    'alexnet': 512,
    'vgg16': 64,
    'inception3': 64,
  }
  model='resnet50'
  run_prefix = 'train'

  cfg = Config(net_name=model, bs=models[model], step_id=10, run_prefix=run_prefix)

  sg = SimGraph(cfg)

if __name__ == "__main__":
  main()