
import os
import copy
import numpy as np

from collections import OrderedDict

import logger
import logging

__all__ = [
  'Node',
  'Graph',
]

class Tensor(object):
  __slots__ = [
    'node_name',
    'slot',
    'allocator_name',
    'alloc_bytes',  # MB
    'alloc_time',
    'dealloc_time',
    'is_temp',
    'is_pers',   # persistent tensor
    'is_alloc',  # if this tensor is allocated (or shared)
    'shared_tensor',
    'share_with', # this tensor is allocated and share with ...
    'out_nodes',
    'last_access',
  ]

  def __init__(self, **kwargs):
    super(Tensor, self).__init__()  # Py2

    self.node_name = None
    self.slot = -1
    self.allocator_name = None
    self.alloc_bytes = 0
    self.alloc_time = -1
    self.dealloc_time = -1
    self.is_temp = False
    self.is_pers = False
    self.is_alloc = False
    self.shared_tensor = None
    self.share_with = []      # if shared_tensor is not None, then this must be empty
    self.out_nodes = []
    self.last_access = None   # which node last accesses this tensor

    for f, v in kwargs.items():
      setattr(self, f, v)

  def __str__(self):
    return '{}\t{}\t{}'.format(self.slot, self.alloc_bytes, self.allocator_name)

  @property
  def name(self):
    return '{}:{}'.format(self.node_name, self.slot)

  @property
  def size(self):
    return self.alloc_bytes

# class ShadowNode(object):
#   __slots__ = [
#     'iter_',
#     'cpu_start_time',
#     'cpu_exec_time',
#     'cpu_end_time',
#     'schedule_time',
#     'temp_allocs',
#     'allocs',
#   ]

#   def __init__(self, **kwargs):
#     super(ShadowNode, self).__init__()  # Py2

#     self.iter_ = -1  # next shadow node's iter will be initialized as iter+1
#     self.cpu_start_time = -1
#     self.cpu_exec_time = -1
#     self.cpu_end_time = -1
#     self.schedule_time = -1
#     self.temp_allocs = []  # temporary allocations
#     self.allocs = []       # allocation in this node (include outputs and temporary memory)

#     for f, v in kwargs.items():
#       setattr(self, f, v)

#   def InitCPUTime(self, node_stat):
#     self.cpu_start_time = node_stat.all_start_micros
#     self.cpu_exec_time = node_stat.all_end_rel_micros
#     self.cpu_end_time = self.cpu_start_time + node_stat.all_end_rel_micros

#   def InitAllocations(self, node_stat):
#     alloc_num = 0
#     for mem in node_stat.memory:
#       if mem.allocator_name.lower() != 'gpu_0_bfc':
#         continue
#       for alloc_rd in mem.allocation_records:
#         alloc_micros = alloc_rd.alloc_micros
#         alloc_bytes = alloc_rd.alloc_bytes
#         self.allocs.append((alloc_micros, alloc_bytes))
#         alloc_num += 1

#     return alloc_num

class Node(object):
  '''
  tf r1.15.2:
  the node time in gpu_stream_all is in kernel-level, which means a node includes
  multiple kernels where each one forms a record
  '''
  __slots__ = [
    'name',
    'iter_',  # a 'while-like' node can have multiple schedule records
    'cpu_start_time',
    'cpu_exec_time',
    'cpu_end_time',
    'gpu_start_time',
    'exec_time',
    'gpu_end_time',
    'schedule_time',
    'inputs',
    'outputs',
    'temp_allocs',
    'allocs',
    'deallocs',
    'kernels_time',  # for debug kernel number
  ]

  def __init__(self, **kwargs):
    super(Node, self).__init__()  # Py2

    self.name = None
    self.iter_ = -1
    # time can not be initialized in constructor as there could be multiple
    # records for a single-node
    self.cpu_start_time = -1 # the schedule time in CPU
    self.cpu_exec_time = -1
    self.cpu_end_time = -1
    self.gpu_start_time = -1
    self.exec_time = -1
    self.gpu_end_time = -1
    self.schedule_time = -1 # self.cpu_start_time - prev_node.cpu_end_time
    self.inputs = dict()   # record input tensors, avoid repeated input tensors
    self.outputs = []      # only includes the outputs of node, not allocations
    self.temp_allocs = []  # temporary allocations
    self.allocs = []       # allocation in this node (include outputs and temporary memory)
    self.deallocs = [] # record which tensor will be released when this node finishes
    self.kernels_time = []

    for f, v in kwargs.items():
      setattr(self, f, v)

    if self.iter_ > 1:  # remain the main node (the first schedule) as convenience
      self.name += '/iter{}'.format(self.iter_)

  def __eq__(self, other):
    return self.cpu_start_time == other.cpu_start_time

  def __gt__(self, other):
    return self.cpu_start_time > other.cpu_start_time

  def __str__(self):
    return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.name, self.cpu_start_time, self.cpu_end_time, self.schedule_time, self.gpu_start_time, self.exec_time)

  def AddInput(self, t):
    if self.inputs.__contains__(t.name):
      logging.info('Repeated input, Node: {}, input: {}'.format(self.name, t.name))
    self.inputs[t.name] = t

  def AddOutput(self, t):
    self.outputs.append(t)

  def SetTime(self):
    # some nodes has no execution time
    if len(self.kernels_time) == 0:
      return

    self.kernels_time.sort(key=lambda x: x[0])
    self.gpu_start_time = self.kernels_time[0][0]

    end_times = [x[0]+x[1] for x in self.kernels_time]
    if max(end_times) != end_times[-1]:
      logging.debug('Not last kernel\'s execution is final: {}'.format(self.name))
    self.gpu_end_time = max(end_times)
    self.exec_time = self.gpu_end_time - self.gpu_start_time

  def InitCPUTime(self, node_stat):
    self.cpu_start_time = node_stat.all_start_micros
    self.cpu_exec_time = node_stat.all_end_rel_micros
    self.cpu_end_time = self.cpu_start_time + node_stat.all_end_rel_micros

  def SetIter(self, it):
    self.iter_ = it

  # def InitShadowNodeCPUTime(self, node_stat):
  #   if len(self.shadow_nodes) == 0:
  #     last_shadow_node = ShadowNode(iter_=1, cpu_start_time=self.cpu_start_time,
  #                                   cpu_exec_time=self.cpu_exec_time,
  #                                   cpu_end_time=self.cpu_end_time,
  #                                   schedule_time=self.schedule_time,
  #                                   temp_allocs=self.temp_allocs,
  #                                   allocs=self.allocs)
  #     # self.shadow_nodes.append(last_shadow_node)
  #     self.shadow_nodes[self.cpu_start_time] = last_shadow_node
    
  #   iter_ = self.shadow_nodes[-1].iter_ + 1
  #   shadow_node = ShadowNode(iter_=iter_)
  #   shadow_node.InitCPUTime(node_stat)
  #   # self.shadow_nodes.append(shadow_node)
  #   self.shadow_nodes[shadow_node.cpu_start_time] = shadow_node

  def InitTime(self, node_stat):
    all_start_micros = node_stat.all_start_micros
    all_end_rel_micros = node_stat.all_end_rel_micros

    self.kernels_time.append((all_start_micros, all_end_rel_micros))

    # if self.start_time == -1:
    #   self.start_time = all_start_micros
    #   self.end_time = self.start_time + all_end_rel_micros
    # else:
    #   try:
    #     assert all_start_micros > self.start_time
    #   except AssertionError:
    #     logging.error('Fail assert following kernel started before the previous: {}'.format(self.name))
    #     logging.error("{} vs {}".format(all_start_micros, self.start_time))
    #     exit(1)
    #   if self.end_time > all_start_micros:
    #     # could be multi-stream, just to see iff exists
    #     logging.info('following kernel started before the previous finished')
    #     logging.info('{}: {}'.format(self.name, self.count))

    #   self.end_time = all_start_micros + all_end_rel_micros

  def InitOutputs(self, node_stat, tensors):
    node_name = node_stat.node_name
    for i in node_stat.output:      
      slot = i.slot

      ad = i.tensor_description.allocation_description
      # alloc_bytes = float(ad.allocated_bytes) / (1<<20)  ## MB
      alloc_bytes = int(ad.allocated_bytes)  ## bytes
      allocator_name = ad.allocator_name
      if allocator_name.lower() != 'gpu_0_bfc':
        continue

      t = Tensor(node_name=node_name,
                 slot=slot,
                 alloc_bytes=alloc_bytes,
                 allocator_name=allocator_name)
      self.outputs.append(t)
      tensors[t.name] = t

  def ProcessOutputs(self):
    slots = np.array([int(t.slot) for t in self.outputs])
    d = dict(zip(*np.unique(slots, return_counts=True)))
    renumber = False
    for v in d.values():
      if v > 1:
        renumber = True 
        # logging.debug("Node [{}] Slots: {}".format(self.name, d))
        break

    if renumber:
      repeated_indices = dict()
      for k in d.keys():
        repeated_indices[k] = 0

      for output in self.outputs:
        temp = output.name.split(':')[-1].strip()
        assert len(temp) == 1
        slot = int(temp)
        curr_index = repeated_indices[slot]
        new_slot = '{}{}'.format(slot, curr_index)
        output.slot = new_slot
        repeated_indices[slot] += 1
    
    return renumber
      

  def InitAllocations(self, node_stat):
    alloc_num = 0
    for mem in node_stat.memory:
      if mem.allocator_name.lower() != 'gpu_0_bfc':
        continue
      for alloc_rd in mem.allocation_records:
        alloc_micros = alloc_rd.alloc_micros
        # alloc_bytes = alloc_rd.alloc_bytes / (1<<20)
        alloc_bytes = alloc_rd.alloc_bytes
        # if alloc_bytes < 0:
        #   continue
        self.allocs.append((alloc_micros, alloc_bytes))
        # self.allocs.append(alloc_bytes)
        alloc_num += 1
    
    return alloc_num

  # def InitShadownNodeAllocations(self, node_stat):
  #   cpu_start_time = node_stat.all_start_micros
  #   assert self.shadow_nodes.__contains__(cpu_start_time)

  #   return self.shadow_nodes[cpu_start_time].InitAllocations(node_stat)

  def UpdateTensorName(self):
    for output in self.outputs:
      output.node_name = self.name
      
    for temp_alloc in self.temp_allocs:
      temp_alloc.node_name = self.name

  def UpdateTensorLastAccess(self, prefix, nodes_map):
    for output in self.outputs:
      if output.last_access is not None:
        if output.last_access.name.startswith('dup1'):
          # already change to duplicated node
          continue
        new_last_access_node_name = prefix + output.last_access.name
        try:
          assert nodes_map.__contains__(new_last_access_node_name)
        except AssertionError:
          logging.error('Can not find duplicated node: {}'.format(new_last_access_node_name))
          exit(1)
        new_last_access_node = nodes_map[new_last_access_node_name]
        output.last_access = new_last_access_node
        # logging.debug('Change Tensor [{}] last access to Node [{}]'.format(output.name, new_last_access_node_name))


class AllocInfo(object):
  def __init__(self, tensor_name, alloc_time, alloc_bytes):
    self.tensor_name = tensor_name
    self.time = alloc_time
    self.bytes = alloc_bytes

  def __eq__(self, other):
    return self.time == other.time and self.bytes == other.bytes and self.tensor_name == other.tensor_name

  def __gt__(self, other):
    if self.time > other.time:
      return True
    elif self.time == other.time:
      if self.bytes < other.bytes:
        return True
      elif self.bytes == other.bytes:
        return self.tensor_name > other.tensor_name
      else:
        return False
    else:
      return False

  def __str__(self):
    return '{}\t{}\t{}'.format(self.tensor_name, self.time, self.bytes)

class Graph():
  def __init__(self, cfg):
    # self.nodes = dict()       # record node time from stream_all, init for each step
    self.nodes = OrderedDict()
    self.shadow_nodes = OrderedDict()  # (node_name: last_iter)
    self.tensors = dict()     # tensors that are the outputs of nodes, shoule not change across iterations
    self.pers_tensors = dict()

    self.peak_mem_tensor_name = None

    self.temp_allocs = []     # temporary allocations, maybe not different across iterations
    self.allocations = []     # record (de)allocations in running

    self.pers_mem = 0
    self.temp_mem = 0

    self.cfg = cfg
    self.alloc_f = '{}/{}{}'.format(cfg.out_dir, cfg.uname+'_'+str(cfg.step_id), '_allocs_gpu_time.log')


  def GetNode(self, node_name):
    if not self.nodes.__contains__(node_name):
      return None

    return self.nodes[node_name]


  def GetOrCreateNode(self, node_name):
    if not self.nodes.__contains__(node_name):
      self.nodes[node_name] = Node(name=node_name)

    return self.nodes[node_name]

  def InitNodeCPUTime(self, nodestats):
    for node_stat in nodestats:
      node_name = node_stat.node_name.split(':')[0]
      if self.GetNode(node_name) is not None:
        # already schedule this node, it should be 'while-like' node
        if not self.shadow_nodes.__contains__(node_name):
          self.shadow_nodes[node_name] = OrderedDict()
          # make last node as a shadow node
          last_iter_node = self.GetNode(node_name)
          last_iter_node.SetIter(1)
          self.shadow_nodes[node_name][last_iter_node.cpu_start_time] = last_iter_node
        curr_iter = len(self.shadow_nodes[node_name]) + 1
        # node = self.GetNode(node_name)
        node = Node(name=node_name, iter_=curr_iter)
        node.InitCPUTime(node_stat)
        assert not self.nodes.__contains__(node.name)
        self.nodes[node.name] = node
        self.shadow_nodes[node_name][node.cpu_start_time] = node
      else:
        node = self.GetOrCreateNode(node_name)
        node.InitCPUTime(node_stat)

    self.InitNodeScheduleTime()
    logging.info('Total nodes (cpu): {}'.format(len(self.nodes)))

  def InitNodeTime(self, nodestats):
    for node_stat in nodestats:      
      node_name = node_stat.node_name.split(':')[0]
      node = self.GetOrCreateNode(node_name)
      node.InitTime(node_stat)

    for node in self.nodes.values():
      node.SetTime()

    logging.info('Total nodes (gpu): {}'.format(len(self.nodes)))
    return len(self.nodes)

  def InitNodeAllocations(self, nodestats):
    total_alloc_num = 0
    for node_stat in nodestats:
      node = self.GetOrCreateNode(node_stat.node_name)
      if node.iter_ != -1:
        assert node.iter_ == 1  # find the main shadow node
        node = self.shadow_nodes[node.name][node_stat.all_start_micros]
      alloc_num = node.InitAllocations(node_stat)
      total_alloc_num += alloc_num

    return total_alloc_num


  def InitNodeOutputs(self, nodestats):
    for node_stat in nodestats:
      # logging.info('InitNodeOutputs: {}'.format(node_stat.node_name))
      node = self.GetOrCreateNode(node_stat.node_name)
      if node.iter_ != -1:
        assert node.iter_ == 1  # find the main shadow node
        node = self.shadow_nodes[node.name][node_stat.all_start_micros]
      node.InitOutputs(node_stat, self.tensors)

    logging.info('Total tensors: {}'.format(len(self.tensors)))
    return len(self.tensors)

    # total_alloc_mem = 0.0
    # for t in self.tensors.values():
    #   total_alloc_mem += t.alloc_bytes

    # logging.info('[InitNodeOutputs] Total allocated memory: {}'.format(total_alloc_mem))

  def CheckShadowNodes(self):
    for main_node_name, shadow_nodes in self.shadow_nodes.items():
      logging.debug('Loop Node [{}] has {} loops'.format(main_node_name, len(shadow_nodes)))
      prev_start_time = -1
      for node in shadow_nodes.values():
        logging.debug('\t{}'.format(node.name))
        assert node.cpu_start_time > prev_start_time
        prev_start_time = node.cpu_start_time

  def ProcessOutputs(self):
    renumber = False
    for node in self.nodes.values():
      res = node.ProcessOutputs()
      if res:
        renumber = True

    if renumber:
      logging.info("Renumber Node outputs")
      with open('{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_node_outputs_renumber.log'), 'w') as fout:
        for node in self.nodes.values():
          if len(node.outputs) == 0:
            continue
          fout.write('{}\n'.format(node.name))
          for t in node.outputs:
            fout.write('{}\n'.format(str(t).strip()))

  def InitNodeScheduleTime(self):
    self.nodes = OrderedDict(sorted(self.nodes.items(), key=lambda x: x[1]))
    assert self.nodes.__contains__('_SOURCE')
    prev_node_cpu_end_time = self.nodes['_SOURCE'].cpu_start_time

    for node in self.nodes.values():
      if node.cpu_start_time == -1:
        continue
      node.schedule_time = node.cpu_start_time - prev_node_cpu_end_time
      prev_node_cpu_end_time = node.cpu_end_time

    # with open('{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname+'_'+str(self.cfg.step_id), '_node_time.log'), 'w') as fout:
    #   for node in self.nodes.values():
    #     fout.write('{}\n'.format(str(node)))

  def InitNodeInputs(self):
    innodes_file = '{}/{}'.format(self.cfg.out_dir, '2_innodes.txt')
    if not os.path.exists(innodes_file):
      logging.error('Can not find innodes file: {}'.format(innodes_file))
      return False

    logging.debug('InitNodeInputs: {}'.format(innodes_file))
    with open(innodes_file) as fin:
      lines = fin.readlines()
      total_length = len(lines)

      i = 0
      while i < total_length:
        tmp = lines[i].split()

        try:
          assert "SrcNode" == tmp[0]
        except AssertionError:
          logging.error("Error line %d with no SrcNode" % i)
          raise AssertionError

        node_name = tmp[1]
        if not self.nodes.__contains__(node_name):
          logging.debug('Can not find node: {}'.format(node_name))
          i = i + int(tmp[2]) + 1
          continue

        pending_count = int(tmp[2])
        node = self.nodes[node_name]
        for j in range(pending_count):
          ttmp = lines[i+j+1].split()
          try:
            assert "InputNode" == ttmp[0]
          except AssertionError:
            logging.error("Error line %d with no InputNode" % (i+j))
            raise AssertionError

          fanin_nodename = ttmp[1]
          fanin_slot = int(ttmp[2])

          if not self.nodes.__contains__(fanin_nodename):
            logging.debug('Can not find fanin node: {}'.format(fanin_nodename))
            # exit(1)
            continue

          if fanin_slot == -1:
            continue  # ignore control flow
          
          t_name = '{}:{}'.format(fanin_nodename, fanin_slot)
          if not self.tensors.__contains__(t_name):
            logging.debug('Can not find tensor: {}'.format(t_name))
            # exit(1)
            continue

          node.AddInput(self.tensors[t_name])
          self.tensors[t_name].out_nodes.append(node)

        i = i + pending_count + 1

    # ordered_tensors = sorted(list(self.tensors.values()), key=lambda x: x.name)
    # with open('./tensor_outnodes.log', 'w') as fout:
    #   for t in ordered_tensors:
    #     fout.write('{}:{}\n'.format(t.name, len(t.out_nodes)))
    return True


  def InitTempAndPersTensors(self):
    # 1. when len(allocs) < len(outputs)
    #   a. the tensor's underlying buffer is shared with another
    #   b. the tensor is persistent allocation (double check!!)
    # 2. when len(allocs) == len(outputs)
    # 3. when len(allocs) > len(outputs)
    #   a. some allocations are temporary memory
    fout_pers = open('{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_pers_allocs.log'), 'w')
    # fout_shared = open('{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_shared.log'), 'w')

    temp_alloc_num = 0

    for node in self.nodes.values():
      # to see which tensor has been allocated memory
      # node_allocs = copy.deepcopy(node.allocs)
      node_allocs = []
      for x in node.allocs:
        if x[1] > 0:
          node_allocs.append(x[1])
      for t in node.outputs:
        if t.size in node_allocs:
          t.is_alloc = True
          node_allocs.remove(t.size)  # prevent single alloc to multi outputs
        else:
          if len(node.inputs) == 0:
            t.is_pers = True
            self.pers_mem += t.alloc_bytes
            fout_pers.write('{}\n'.format(t.name))

      # the left allocs in node.allocs are temporary tensors
      for alloc in node_allocs:
        t = Tensor(node_name=node.name, alloc_bytes=alloc, is_temp=True)
        node.temp_allocs.append(t)
        self.temp_allocs.append(t)
        self.temp_mem += alloc

    # shared_tensors = [t for t in self.tensors.values() if not t.is_alloc and not t.is_pers]
    # for t in shared_tensors:
    #   if t.shared_tensor != None:
    #     continue

    #   node = self.GetNode(t.node_name)
    #   curr_node = node
    #   shared_chain = []
    #   shared_chain.append(t)
    #   while True:
    #     index = -1
    #     # find the first input with the same size
    #     for i, it in enumerate(curr_node.inputs.values()):
    #       if it.size == t.size:
    #         index = i
    #         break
        
    #     if index == -1:
    #       logging.error('Can not find same size input: {}'.format(t.name))
    #       exit(1)

    #     curr_t = list(curr_node.inputs.values())[index]
    #     # if find a tensor whose shared tensor is already set
    #     if curr_t.shared_tensor != None:
    #       for st in shared_chain:
    #         st.shared_tensor = curr_t.shared_tensor
    #       break

    #     if not curr_t.is_pers and not curr_t.is_alloc:
    #       # no allocation in curr_t, find recursive
    #       curr_node = self.GetNode(curr_t.node_name)
    #       shared_chain.append(curr_t)
    #     else:
    #       # set all tensor in this chain
    #       for st in shared_chain:
    #         st.shared_tensor = curr_t
    #       break

    # for t in shared_tensors:
    #   t.shared_tensor.share_with.append(t)
    #   t.shared_tensor.out_nodes += t.out_nodes

    # for t in self.tensors.values():
    #   if len(t.share_with) == 0:
    #     continue

    #   assert t.shared_tensor == None
    #   fout_shared.write('{}\n'.format(t.name))
    #   for tt in t.share_with:
    #     fout_shared.write('\t{}\n'.format(tt.name))
        
    # fout_pers.close()
    # fout_shared.close()

    logging.info('Total temporary memory size: {}'.format(self.temp_mem))
    return temp_alloc_num


  def DetermineTensorDealloc(self):

    def BinarySearch(array, l, r, t):
      if l >= r:
        return l

      mid = l + (r-l)//2

      if array[mid][2] == t:
        return mid

      num_mid = array[mid][2]
      num_left = array[l if (mid-1) < l else (mid-1)][2]
      num_right = array[r if (mid+1) > r else (mid+1)][2]
      sm = abs(t-num_mid)
      sl = abs(t-num_left)
      sr = abs(t-num_right)

      if (sm < sl and sm < sr):
        return mid
      else:
        l_index = BinarySearch(array, l, mid-1, t)
        r_index = BinarySearch(array, mid+1, r, t)
        if (abs(t-array[l_index][2]) < abs(t-array[r_index][2])):
          return l_index
        else:
          return r_index

    def FindOutputBytes(bytes, outputs):
      id_ = -1
      for i in range(len(outputs)-1, -1, -1):
        if outputs[i][1] == abs(bytes):
          id_ = i
          break

      if id_ == -1:
        return -1
      else:
        res = outputs[id_][0]
        del outputs[id_]
        return res
        

    ordered_nodes_time = [(node.name, node.cpu_start_time, node.cpu_end_time) for node in self.nodes.values()]
    ordered_nodes_time.sort(key=lambda x: x[2])
    for node in self.nodes.values():
      alloc_num = len(node.allocs)
      if alloc_num == 0:
        continue
      try:
        assert alloc_num % 2 == 0
      except AssertionError:
        logging.error("Node [{}] allocation number is not even number! Alloc : {}".format(node.name, node.allocs))
        # exit(1)
        continue

    
      # outputs_info = [(i, t.size) for i, t in enumerate(node.outputs)]
      outputs_info = []
      # not all allocation is listed before deallocation
      alloc_info = []
      dealloc_info = []
      for micros, bytes in node.allocs:
        if bytes > 0:
          alloc_info.append((micros, bytes))
        else:
          dealloc_info.append((micros, bytes))
      alloc_sizes = [x for _, x in alloc_info]

      # to identify which tensor own the underlying memory
      for oi, output in enumerate(node.outputs):
        try:
          i = alloc_sizes.index(output.size)
        except ValueError:
          # logging.error('Can not find Tensor [{}] Size [{}] allocation info in {}'.format(output.name, output.size, str(alloc_sizes)))
          if len(node.inputs) == 0:
            output.is_pers = True  # TODO(px): seems this can not identify which tensor is persistent tensor
          else:
            pass
            # logging.debug('Tensor [{}] is a shared tensor'.format(output.name))
          continue
        del alloc_sizes[i]
        output.is_alloc = True
        outputs_info.append((oi, output.size))
        # logging.debug("Tensor [{}] alloc is True".format(output.name))

      temp_alloc_index = 1
      temp_weird_alloc_index = 1
      for micros, bytes in dealloc_info:
        # find corresponding tensor first
        output_index = FindOutputBytes(bytes, outputs_info)
        # logging.debug("Find output_index: {} for dealloc: {} at Node {}".format(output_index, bytes, node.name))
        if output_index == -1:
          # is a temporary allocation (NOTE: but may not be released immediately at this node)
          if node.cpu_start_time < micros <= node.cpu_end_time:
            t_tensor = Tensor(node_name=node.name, slot='t{}'.format(temp_alloc_index), alloc_bytes=abs(bytes), is_temp=True)
            temp_alloc_index += 1
          else:
            t_tensor = Tensor(node_name=node.name, slot='w{}'.format(temp_weird_alloc_index), alloc_bytes=abs(bytes), is_temp=True)
            temp_weird_alloc_index += 1
          node.temp_allocs.append(t_tensor)
          self.tensors[t_tensor.name] = t_tensor
            # logging.debug("Find a weird temporary allocation: {}".format(t_tensor.name))
        else:
          t_tensor = node.outputs[output_index]
          # logging.debug("Find {}".format(t_tensor.name))

        t_node = None  # at which node, this allocation will be released
        if node.cpu_start_time < micros <= node.cpu_end_time:
          t_node = node
        else:
          index = BinarySearch(ordered_nodes_time, 0, len(ordered_nodes_time)-1, micros)
          if index == -1:
            logging.error('Can not find which node will release tensor in Node [{}], size: [{}], deallocated in {}'.format(node.name, bytes, micros))
            exit(1)
          t_node = self.nodes[ordered_nodes_time[index][0]]

        t_tensor.last_access = t_node
        logging.debug('Tensor [{}, {}] find t_node {}'.format(t_tensor.name, t_tensor.size, t_node.name))

      # has_temp = len(alloc_sizes)
      # # left allocations are temporary allocation
      # # TODO(px): the left may not be temp allocations (in deepspeech2 and tacotron)
      # if len(alloc_sizes) != 0:
      #   # logging.debug('Node [{}] has {} temporary allocations'.format(node.name, len(alloc_sizes)))
      #   for i, alloc_size in enumerate(alloc_sizes):
      #     t = Tensor(node_name=node.name, slot='t{}'.format(i), alloc_bytes=alloc_size, is_temp=True)
      #     node.temp_allocs.append(t)


      # for micros, bytes in dealloc_info:
      #   t_node = None
      #   # whether it's a temp mem (DOUBLE-CHECK: a normal tensor is generated at this node and release at this node?)
      #   if node.cpu_start_time < micros <= node.cpu_end_time and abs(bytes) in alloc_sizes:
      #     if has_temp:
      #       # is a temp deallocation
      #       has_temp -= 1
      #       continue
      #     else:
      #       t_node = node

      #   if t_node is None:
      #     # find the dealloc time is in which node's scheduling (cpu)
      #     index = BinarySearch(ordered_nodes_time, 0, len(ordered_nodes_time)-1, micros)
      #     # logging.debug('Find {} for [{}]\'s dealloc: {}, {}'.format(index, node.name, micros, bytes))
      #     if index == -1:
      #       logging.error('Can not find which node will release tensor in Node [{}], size: [{}], deallocated in {}'.format(node.name, bytes, micros))
      #       exit(1)
      #     t_node = self.nodes[ordered_nodes_time[index][0]]

      #   # find the corresponding alloc info in outputs
      #   id_ = -1
      #   for i in range(len(outputs_info)-1, -1, -1):
      #     if outputs_info[i][1] == abs(bytes):
      #       id_ = i
      #       break
      #   if id_ == -1:
      #     logging.error('Can not find the corresponding output tensor: Node [{}] Dealloc size {}, outputs_info: {}'.format(node.name, bytes, str(outputs_info)))

      #   output_index = outputs_info[id_][0]
      #   t_tensor = node.outputs[output_index]
      #   t_node.deallocs.append(t_tensor)
      #   assert t_tensor.is_alloc
      #   t_tensor.last_access = t_node
      #   # logging.debug('Tensor [{}, {}] find t_node {}'.format(t_tensor.name, t_tensor.size, t_node.name))
      #   del outputs_info[id_]

    # debug log
    # for node in self.nodes.values():
    #   for output in node.outputs:
    #     if output.is_pers:
    #       continue
    #     if not output.is_alloc:
    #       continue
    #     if output.last_access is None:
    #       logging.error('Tensor [{}] last access node is None!'.format(output.name))
    #       continue

  def CalculatePeakMem(self, log_alloc_trace=False):
    '''
    Estimate each tensor's allocation time and deallocation time from node.cpu_start_time and node.cpu_end_time.
    And calculate peak memory using this estimated allocations information.
    '''
    self.allocations = []

    for node in self.nodes.values():
      for output in node.outputs:
        if output.is_alloc:
          self.allocations.append((node.cpu_start_time, output.name, output.size))
          # logging.debug('[Tensor {}, Size {}, Alloc time {}]'.format(output.name, output.size, node.cpu_start_time))
          try:
            assert output.last_access
          except AssertionError:
            logging.error('Tensor [{}, {}] ({}, ) last access is None'.format(output.name, output.size, node.cpu_start_time))
            exit(1)
          self.allocations.append((output.last_access.cpu_end_time, output.name, -output.size))
          # logging.debug('[Tensor {}, Dealloc time ({}, {}), Dealloc node: {}'.format(output.name, node.cpu_start_time, output.last_access.cpu_end_time, output.last_access.name))
        else:
          pass

      for t in node.temp_allocs:
        assert t.last_access
        self.allocations.append((node.cpu_start_time, t.name, t.size))
        self.allocations.append((t.last_access.cpu_end_time, t.name, -t.size))
        # self.allocations.append((node.cpu_start_time, t.name, t.size))
        # self.allocations.append((node.cpu_end_time, t.name, -t.size))

    self.allocations.sort(key=lambda x : x[0])
    if log_alloc_trace:
      with open(self.cfg.out_dir+'/alloc_trace.log', 'w') as fout:
        for micros, name, bytes in self.allocations:
          fout.write('{} {} {}\n'.format(micros, name, bytes))
          
    peak_mem = 0.0
    peak_mem_micros = 0.0
    curr_mem = 0.0
    for micros, tensor_name, bytes in self.allocations:
      curr_mem += bytes / (1<<20)  # in MB
      if curr_mem > peak_mem:
        peak_mem = curr_mem
        peak_mem_micros = micros
        self.peak_mem_tensor_name = tensor_name
    
    all_start_micros = self.nodes['_SOURCE'].cpu_start_time
    total_schedule_time = list(self.nodes.values())[-1].cpu_end_time - all_start_micros
    logging.info('Total allocations number: {}, calculate peak memory: {} MB'.format(len(self.allocations), peak_mem))
    logging.info('Peak memory tensor: {}, Peak memory micros {}/{}'.format(self.peak_mem_tensor_name, peak_mem_micros-all_start_micros, total_schedule_time))

  def LogAlloc(self, filename=None):
    if filename is None:
      filename = self.cfg.out_dir+'/'+self.cfg.uname+'_alloc_trace.log'
    with open(filename, 'w') as fout:
      for micros, name, bytes in self.allocations:
        fout.write('{} {} {}\n'.format(micros, name, bytes))


  def AccurateMemUsage(self):
    '''
    Get accurate peak memory from allocations log from tf.run_metadata
    '''
    allocations = []
    for node in self.nodes.values():
      for alloc in node.allocs:
        allocations.append((alloc[0], alloc[1]))

    allocations.sort(key=lambda x: x[0])
    curr_mem = 0
    peak_mem = 0
    for _, alloc_bytes in allocations:
      curr_mem += alloc_bytes / (1<<20)  # in MB
      if curr_mem > peak_mem:
        peak_mem = curr_mem

    logging.info('Total allocations: {}, Accurate peak memory: {} MB'.format(len(allocations), peak_mem))


  def InitSharedTensors(self):
    shared_tensors = [t for t in self.tensors.values() if not t.is_alloc and not t.is_pers]
    for t in shared_tensors:
      if t.shared_tensor != None:
        continue

      node = self.GetNode(t.node_name)
      curr_node = node
      shared_chain = []
      shared_chain.append(t)

      while True:
        index = -1
        # find the first input with the same size
        for i, it in enumerate(curr_node.inputs.values()):
          if it.size == t.size:
            index = i
            break
        
        if index == -1:
          logging.error('Can not find same size input for {} in {}'.format(t.name, curr_node.name))
          exit(1)

        curr_t = list(curr_node.inputs.values())[index]
        # if find a tensor whose shared tensor is already set
        if curr_t.shared_tensor != None:
          for st in shared_chain:
            st.shared_tensor = curr_t.shared_tensor
          break

        if not curr_t.is_pers and not curr_t.is_alloc:
          # no allocation in curr_t, find recursive
          curr_node = self.GetNode(curr_t.node_name)
          shared_chain.append(curr_t)
        else:
          # set all tensor in this chain
          for st in shared_chain:
            st.shared_tensor = curr_t
          break

    for t in shared_tensors:
      t.shared_tensor.share_with.append(t)
      t.shared_tensor.out_nodes += t.out_nodes


  # Get memory usage using GPU time
  def GetMemUsage(self, plot_func=None):
    self.allocations = []
    # outputs tensors
    for t in self.tensors.values():
      if t.is_pers:
        continue   # pass persistent tensor for now
      if not t.is_alloc:
        assert t.shared_tensor != None
        continue

      node = self.GetNode(t.node_name)
      t.alloc_time = node.cpu_start_time
      dealloc_time = node.cpu_end_time
      for tn in t.out_nodes:
        if tn.cpu_end_time > dealloc_time:
          dealloc_time = tn.cpu_end_time

      t.dealloc_time = dealloc_time
      self.allocations.append((t.alloc_time, t.size))
      self.allocations.append((t.dealloc_time, -t.size))

    for t in self.temp_allocs:
      node = self.GetNode(t.node_name)
      self.allocations.append((node.cpu_start_time, t.size))
      self.allocations.append((node.cpu_end_time, -t.size))

    self.allocations.sort(key=lambda x: x[0])
    with open(self.alloc_f, 'w') as fout:
      for t, b in self.allocations:
        fout.write('{}\t{}\n'.format(t, b))    
    

    
    start = self.allocations[0][0]
    curr_mem = 0.0
    max_mem = 0.0
    x = []
    y = []

    for data in self.allocations:
      t = data[0] - start
      curr_mem = curr_mem + float(data[1]) / (1 << 10)  # GB
      x.append(t)
      y.append(curr_mem)
      if curr_mem > max_mem:
        max_mem = curr_mem

    logging.info('Peak memory: {}\t{}(w/ persistent memory)'.format(max_mem, max_mem+self.pers_mem/(1<<10)))
    if plot_func:
      plot_func(self.cfg.uname, x=x, y=y, fig_dir=self.cfg.figure_dir)



  def InitFromFile(self):
    if self.cfg.step_id == -1:
      node_time_f = '{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_node_time.log')
      node_allocs_f = '{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_node_allocs.log')
    else:
      node_time_f = '{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname+'_'+str(self.cfg.step_id), '_node_time.log')
      node_allocs_f = '{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname+'_'+str(self.cfg.step_id), '_node_allocs.log')

    renumber_outputs_f = '{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_node_outputs_renumber.log')
    if os.path.exists(renumber_outputs_f):
      node_outputs_f = renumber_outputs_f
    else:
      node_outputs_f = '{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_node_outputs.log')
    assert os.path.exists(node_time_f)
    assert os.path.exists(node_allocs_f)
    assert os.path.exists(node_outputs_f)

    logging.debug('Init Node time from: {}'.format(node_time_f))
    logging.debug('Init Node allocs from: {}'.format(node_allocs_f))
    logging.debug('Init Node outputs from: {}'.format(node_outputs_f))

    # Init node time
    with open(node_time_f) as fin:
      for line in fin:
        temp = line.split('\t')
        node = self.GetOrCreateNode(node_name=temp[0])
        node.cpu_start_time = int(temp[1])
        node.cpu_end_time = int(temp[2])
        node.cpu_exec_time = node.cpu_end_time - node.cpu_start_time
        node.schedule_time = int(temp[3])
        node.gpu_start_time = int(temp[4])
        node.exec_time = int(temp[5])
        node.gpu_end_time = node.gpu_start_time+node.exec_time

      logging.info('Total Nodes: {}'.format(len(self.nodes)))

    with open(node_allocs_f) as fin:
      curr_node = None
      for line in fin:
        # if line.startswith('\t'):
        if line[0].isdigit():
          # logging.info(line[2:-2])
          # tmp = line[2:-2].split(',')
          # alloc_bytes = float(tmp[1].strip())
          temp = line.split('\t')
          alloc_micros = int(temp[0])
          # alloc_bytes = float(temp[1].strip())
          alloc_bytes = int(temp[1].strip())
          assert curr_node
          curr_node.allocs.append((alloc_micros, alloc_bytes))
        else:
          curr_node = self.GetOrCreateNode(line.strip())

    with open(node_outputs_f) as fin:
      curr_node = None
      for line in fin:
        if line[0].isdigit():
          tmp = line.split('\t')
          slot = tmp[0]
          # alloc_bytes = float(tmp[1])
          alloc_bytes = int(tmp[1])
          allocator_name = tmp[2]
          
          assert curr_node
          t = Tensor(node_name=curr_node.name, slot=slot,
                     alloc_bytes=alloc_bytes, allocator_name=allocator_name)
          curr_node.AddOutput(t)
          self.tensors[t.name] = t
        else:
          curr_node = self.GetOrCreateNode(line.strip())

      logging.info('Total tensors: {}'.format(len(self.tensors)))

    tensors_dealloc_f = '{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_tensors_dealloc.log')
    if os.path.exists(tensors_dealloc_f):
      logging.info("Init Tensor Deallocation information from log")
      with open(tensors_dealloc_f) as fin:
        curr_node = None
        for line in fin:
          if line[0] == '\t':
            assert curr_node
            tmp = line[1:].strip().split('\t')
            slot = tmp[1]
            last_access_name = tmp[2]
            try:
              assert self.nodes.__contains__(last_access_name)
            except AssertionError:
              logging.error("Can not find last_access node name: {}".format(line))
              exit(1)
            if tmp[0] == 'normal':
              tensor_name = '{}:{}'.format(curr_node.name, slot)
              assert self.tensors.__contains__(tensor_name)
              t = self.tensors[tensor_name]
              t.is_alloc = True
              
              t.last_access = self.nodes[last_access_name]
              logging.debug("Tensor [{}] last access is: {}".format(tensor_name, last_access_name))
            elif tmp[0] == 'temp':
              assert slot[0] == 't' or slot[0] == 'w'
              tensor_name = '{}:{}'.format(curr_node.name, slot)
              assert not self.tensors.__contains__(tensor_name)
              t = Tensor(node_name=curr_node.name, slot=slot, alloc_bytes=int(tmp[3]), is_temp=True)
              t.last_access = self.nodes[last_access_name]
              curr_node.temp_allocs.append(t)
              self.tensors[t.name] = t
            else:
              logging.error("Error line: {}".format(line))
              exit(1)

          else:
            node_name = line.strip()
            assert self.nodes.__contains__(node_name)
            curr_node = self.nodes[node_name]
    else:
      self.DetermineTensorDealloc()
      with open(tensors_dealloc_f, 'w') as fout:
        for node in self.nodes.values():
          fout.write('{}\n'.format(node.name))
          for output in node.outputs:
            if output.is_alloc:
              fout.write('\tnormal\t{}\t{}\n'.format(output.slot, output.last_access.name))
          for t in node.temp_allocs:
            fout.write('\ttemp\t{}\t{}\t{}\n'.format(t.slot, t.last_access.name, t.size))
        
    # self.InitNodeInputs()
    # self.InitTempAndPersTensors()
    # self.InitSharedTensors()
    # self.ProcessOutputs()


  # for debug
  # def diff(self, cfg):
  #   time_only = set(self.nodes.keys()).difference(self.nodes1.keys())
  #   output_only = set(self.nodes1.keys()).difference(self.nodes.keys())

  #   with open('{}/{}{}'.format(cfg.out_dir, cfg.uname, '_nodediff.log'), 'w') as fout:
  #     fout.write('Nodes with only time:\n')
  #     for n in time_only:
  #       fout.write('{}\n'.format(n))
      
  #     fout.write('Nodes with only output:\n')
  #     for n in output_only:
  #       fout.write('{}\n'.format(n))

  # def _debug_node_wo_time(self, cfg):
  #   with open('{}/{}{}'.format(cfg.out_dir, cfg.uname, '_node_wo_time.log'), 'w') as fout:
  #     for node in self.nodes.values():
  #       if node.start_time != -1:
  #         continue

  #       if len(node.allocs) == 0:
  #         continue
  #       fout.write('{}\n'.format(node.name))
  #       for alloc in node.allocs:
  #         fout.write('\t{}\n'.format(alloc))

  def DumpToFile(self):
    logging.info('Dump info to file...')
    self.nodes = OrderedDict(sorted(self.nodes.items(), key=lambda x: x[1]))
    with open('{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname+'_'+str(self.cfg.step_id), '_node_time.log'), 'w') as fout:
      for node in self.nodes.values():
        fout.write('{}\n'.format(node))

    with open('{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_node_outputs.log'), 'w') as fout:
      for node in self.nodes.values():
        if len(node.outputs) == 0:
          continue
        fout.write('{}\n'.format(node.name))
        for t in node.outputs:
          fout.write('{}\n'.format(t))

    # as node.allocs include alloc micros, thus add step_id to identify
    with open('{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname+'_'+str(self.cfg.step_id), '_node_allocs.log'), 'w') as fout:
      for node in self.nodes.values():
        fout.write('{}\n'.format(node.name))
        for alloc in node.allocs:
          fout.write('{}\t{}\n'.format(alloc[0], alloc[1]))

    # with open('{}/{}{}'.format(self.cfg.out_dir, self.cfg.uname, '_temp_allocs.log'), 'w') as fout:
    #   for node in self.nodes.values():
    #     fout.write('{}\n'.format(node.name))
    #     for tmp_alloc in node.temp_allocs:
    #       fout.write('\t{}\n'.format(tmp_alloc))
