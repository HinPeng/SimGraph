from __future__ import print_function

import copy
import os
import logger
import logging

from collections import OrderedDict
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
    # self.graph.LogAlloc()
    self.graph.AccurateMemUsage()
    # self.graph.GetMemUsage()

  def LogAlloc(self, filename=None):
    return self.graph.LogAlloc(filename=filename)

  def SimDuplicatedJobsSchedule(self, duplicated_num=1):
    self.graph.nodes = OrderedDict(sorted(self.graph.nodes.items(), key=lambda x: x[1]))
    prev_node_cpu_end_time = self.graph.nodes['_SOURCE'].cpu_start_time

    duplicated_nodes = OrderedDict()

    prefix = 'dup{}/'.format(duplicated_num)
    for node in self.graph.nodes.values():
      if node.cpu_start_time == -1:
        continue
      
      node.cpu_start_time = prev_node_cpu_end_time + node.schedule_time
      node.cpu_end_time = node.cpu_start_time + node.cpu_exec_time
      prev_node_cpu_end_time = node.cpu_end_time

      
      dup_node = copy.deepcopy(node)
      dup_node.name = prefix + node.name
      dup_node.cpu_start_time = prev_node_cpu_end_time + dup_node.schedule_time
      dup_node.cpu_end_time = dup_node.cpu_start_time + dup_node.cpu_exec_time
      prev_node_cpu_end_time = dup_node.cpu_end_time

      dup_node.UpdateTensorName()
      duplicated_nodes[dup_node.name] = dup_node

    for name, node in duplicated_nodes.items():
      # logging.debug('Add [{}, ({}, {})]'.format(name, node.cpu_start_time, node.cpu_end_time))
      node.UpdateTensorLastAccess(prefix, duplicated_nodes)
      self.graph.nodes[name] = node

    self.graph.CalculatePeakMem()
    
    

def main():
  models = {
    'resnet50': 64,
    'resnet152': 64,
    'alexnet': 512,
    'vgg16': 64,
    'inception3': 64,
    'inception4': 64,
    'Bert': 32,
    'Tacotron': 32,
    'deepspeech2': 32,
  }
  
  run_prefix = 'train'
  # model='Tacotron'
  # step_id = 30

  model='resnet50'
  step_id = 10

  cfg = Config(net_name=model, bs=models[model], step_id=step_id, run_prefix=run_prefix)

  sg = SimGraph(cfg)
  # sg.SimDuplicatedJobsSchedule(1)
  # filename = cfg.out_dir+'/'+cfg.uname+'_dup2_alloc_trace.log'
  # sg.LogAlloc(filename)

if __name__ == "__main__":
  main()