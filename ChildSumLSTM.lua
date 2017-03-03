--[[
The MIT License (MIT)

Copyright (c) 2016 Justin Johnson
--]]

--[[
Thank you Justin for this awesome super fast code: 
 * https://github.com/jcjohnson/torch-rnn

If we add up the sizes of all the tensors for output, gradInput, weights,
gradWeights, and temporary buffers, we get that a SeqLSTM stores this many
scalar values:

NTD + 6NTH + 8NH + 8H^2 + 8DH + 9H

N : batchsize; T : seqlen; D : inputsize; H : outputsize

For N = 100, D = 512, T = 100, H = 1024 and with 4 bytes per number, this comes
out to 305MB. Note that this class doesn't own input or gradOutput, so you'll
see a bit higher memory usage in practice.
--]]
local ChildSumLSTM, parent = torch.class('nn.ChildSumLSTM', 'nn.Module')

function ChildSumLSTM:__init(inputsize, hiddensize, outputsize)
   parent.__init(self)
   -- for non-SeqLSTMP, only inputsize, hiddensize=outputsize are provided
   outputsize = outputsize or hiddensize
   local D, H, R = inputsize, hiddensize, outputsize
   self.inputsize, self.hiddensize, self.outputsize = D, H, R
   
   self.weight = torch.Tensor(D+R, 4 * H) -- order [i o g f]
   self.gradWeight = torch.Tensor(D+R, 4 * H) -- order [i o g f]

   self.bias = torch.Tensor(4 * H) -- order [i o g f]
   self.gradBias = torch.Tensor(4 * H):zero() -- order [i o g f]
   self:reset()

   self.cell = torch.Tensor()    -- This will be  (T, N, H)
   self.gates = torch.Tensor()   -- This will be (T, N, 3H)
   self.fgates = torch.Tensor() -- This will be (T, N, T, H)
   self.buffer0 = torch.Tensor() -- This will be (N, T, H)
   self.buffer1 = torch.Tensor() -- This will be (N, H)
   self.buffer2 = torch.Tensor() -- This will be (N, H)
   self.buffer3 = torch.Tensor() -- This will be (1, 4H)
   self.grad_a_buffer = torch.Tensor() -- This will be (N, 3H)
   self.grad_f_buffer = torch.Tensor() -- This will be (N, T, H)

   self.h0 = torch.Tensor()
   self.c0 = torch.Tensor()

   self._remember = 'neither'

   self.grad_c0 = torch.Tensor()
   self.grad_h0 = torch.Tensor()
   self.grad_x = torch.Tensor()
   self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
   
   -- set this to true to forward inputs as batchsize x seqlen x ...
   -- instead of seqlen x batchsize
   self.batchfirst = false
   -- set this to true for variable length sequences that seperate
   -- independent sequences with a step of zeros (a tensor of size D)
   self.maskzero = false
end

function ChildSumLSTM:reset(std)
   if not std then
      std = 1.0 / math.sqrt(self.outputsize + self.inputsize)
   end
   self.bias:zero()
   self.bias[{{self.outputsize + 1, 2 * self.outputsize}}]:fill(1)
   self.weight:normal(0, std)
   return self
end

function ChildSumLSTM:resetStates()
   self.h0 = self.h0.new()
   self.c0 = self.c0.new()
end

-- unlike MaskZero, the mask is applied in-place
function ChildSumLSTM:recursiveMask(output, mask)
   if torch.type(output) == 'table' then
      for k,v in ipairs(output) do
         self:recursiveMask(output[k], mask)
      end
   else
      assert(torch.isTensor(output))
      
      -- make sure mask has the same dimension as the output tensor
      local outputSize = output:size():fill(1)
      outputSize[1] = output:size(1)
      mask:resize(outputSize)
      -- build mask
      local zeroMask = mask:expandAs(output)
      output:maskedFill(zeroMask, 0)
   end
end

local function check_dims(x, dims)
   assert(x:dim() == #dims)
   for i, d in ipairs(dims) do
      assert(x:size(i) == d)
   end
end

-- makes sure x, h0, c0 and gradOutput have correct sizes.
-- batchfirst = true will transpose the N x T to conform to T x N
function ChildSumLSTM:_prepare_size(input, gradOutput)
   local c0, h0, x
   if torch.type(input) == 'table' and #input == 3 then
      c0, h0, x = unpack(input)
   elseif torch.type(input) == 'table' and #input == 2 then
      h0, x = unpack(input)
   elseif torch.isTensor(input) then
      x = input
   else
      assert(false, 'invalid input')
   end
   assert(x:dim() == 3, "Only supports batch mode")
   
   if self.batchfirst then
      x = x:transpose(1,2)
      gradOutput = gradOutput and gradOutput:transpose(1,2) or nil
   end
   
   local T, N = x:size(1), x:size(2)
   local H, D = self.outputsize, self.inputsize
   
   check_dims(x, {T, N, D})
   if h0 then
      check_dims(h0, {N, H})
   end
   if c0 then
      check_dims(c0, {N, H})
   end
   if gradOutput then
      check_dims(gradOutput, {T, N, H})
   end
   return c0, h0, x, gradOutput
end

--[[
Input: {{c0, h0, x}/{h0,x}/x, pred}
- c0: Initial cell state, (N, H)
- h0: Initial hidden state, (N, H)
- x: Input sequence, (T, N, D)  

Output:
- h: Sequence of hidden states, (T, N, H)
--]]

function ChildSumLSTM:updateOutput(input)
   self.recompute_backward = true
   local c0, h0, x = self:_prepare_size(input[1])
   local N, T = x:size(2), x:size(1)
   self.hiddensize = self.hiddensize or self.outputsize -- backwards compat
   local H, R, D = self.hiddensize, self.outputsize, self.inputsize
   
   self._output = self._output or self.weight.new()
   
   -- remember previous state?
   local remember
   if self.train ~= false then -- training
      if self._remember == 'both' or self._remember == 'train' then
         remember = true
      elseif self._remember == 'neither' or self._remember == 'eval' then
         remember = false
      end
   else -- evaluate
      if self._remember == 'both' or self._remember == 'eval' then
         remember = true
      elseif self._remember == 'neither' or self._remember == 'train' then
         remember = false
      end
   end

   self._return_grad_c0 = nil -- (c0 ~= nil)
   self._return_grad_h0 = nil -- (h0 ~= nil)

   local bias_expand = self.bias:narrow(1, 1, 3 * H):view(1, 3 * H):expand(N, 3 * H)
   local fbias_expand = self.bias:narrow(1, 3 * H + 1, H):view(1, H):expand(N, H)
   local connect_expand = input[2]:view(T, N, T, 1):expand(T, N, T, H)
   local Wx = self.weight:narrow(1,1,D):narrow(2, 1, 3 * H)
   local Wh = self.weight:narrow(1,D+1,R):narrow(2, 1, 3 * H)
   local Wfx = self.weight:narrow(1, 1, D):narrow(2, 3 * H + 1, H)
   local Wfh = self.weight:narrow(1, D + 1, R):narrow(2, 3 * H + 1, H)

   local h, c = self._output, self.cell
   h:resize(T, N, R):zero()
   c:resize(T, N, H):zero()
   -- local prev_h, prev_c = h0, c0
   self.gates:resize(T, N, 3 * H):zero()
   self.fgates:resize(T, N, T, H):zero()
   self.buffer0:resize(N, T, H):zero()
   self.buffer1:resize(N, H):zero()
   for t = 1, T do
      -- Collect predecessors' sum
      local cur_x = x[t]
      local cur_h = h[t]
      local cur_c = c[t]
      local cur_gates = self.gates[t] -- size: (N, H)
      local cur_fgates = self.fgates[t] -- size: (N, T, H)
      cur_gates:addmm(bias_expand, cur_x, Wx)
      self.buffer0:cmul(h:transpose(1, 2), connect_expand[t]) -- size: (N, T, H)
      self.buffer1:copy(torch.sum(self.buffer0, 2):view(N, H)) -- size: (N, H)
      --
      cur_gates:addmm(self.buffer1, Wh)
      cur_gates[{{}, {1, 2 * H}}]:sigmoid() -- i,o gate
      cur_gates[{{}, {2 * H + 1, 3 * H}}]:tanh() -- g gate

      self.buffer1:zero():addmm(fbias_expand, cur_x, Wfx)
      cur_fgates:add(self.buffer1:view(N, 1, H):expand(N, T, H))
      for j = 1, N do
         cur_fgates[j]:addmm(self.buffer0[j], Wfh) -- size: (T, H)
      end
      cur_fgates:sigmoid() -- f gate size: (N, T, H)

      local i = cur_gates[{{}, {1, H}}] -- input gate
      local o = cur_gates[{{}, {H + 1, 2 * H}}] -- output gate
      local g = cur_gates[{{}, {2 * H + 1, 3 * H}}] -- input transform
      local f = cur_fgates -- f gate size: (N, T, H)

      self.buffer0:cmul(c:transpose(1, 2), connect_expand[t]):cmul(f) -- sum(f * c) size: (N, T, H)
      cur_c:add(self.buffer0:sum(2):view(N, H)):addcmul(i, g) -- size: (N, H)
      cur_h:tanh(cur_c):cmul(o) -- size: (N, H)
      
      -- for LSTMP
      self:adapter(t)
      
      if self.maskzero then
         -- build mask from input
         local vectorDim = cur_x:dim() 
         self._zeroMask = self._zeroMask or cur_x.new()
         self._zeroMask:norm(cur_x, 2, vectorDim)
         self.zeroMask = self.zeroMask or ((torch.type(cur_x) == 'torch.CudaTensor') and torch.CudaByteTensor() or torch.ByteTensor())
         self._zeroMask.eq(self.zeroMask, self._zeroMask, 0)     
         -- zero masked output
         self:recursiveMask({self.next_h, next_c, cur_gates}, self.zeroMask)
      end
      
      -- prev_h, prev_c = self.next_h, next_c
   end
   self.userPrevOutput = nil
   self.userPrevCell = nil
   
   if self.batchfirst then
      self.output = self._output:transpose(1,2) -- T x N -> N X T
   else
      self.output = self._output
   end

   return self.output
end

function ChildSumLSTM:adapter(scale, t)
   -- Placeholder for ChildSumLSTMP
end

function ChildSumLSTM:backward(input, gradOutput, scale)
   self.recompute_backward = false
   scale = scale or 1.0
   assert(scale == 1.0, 'must have scale=1')
   
   local c0, h0, x, grad_out_h = self:_prepare_size(input[1], gradOutput)
   assert(grad_out_h, "Expecting gradOutput")
   local N, T = x:size(2), x:size(1)
   self.hiddensize = self.hiddensize or self.outputsize -- backwards compat
   local H, R, D = self.hiddensize, self.outputsize, self.inputsize
   local connect_expand = input[2]:view(T, N, T, 1):expand(T, N, T, H)
   
   self._grad_x = self._grad_x or self.weight:narrow(1,1,D).new()
   self._grad_c = self._grad_c or self.cell.new() -- size: (T, N, H)
   self._grad_h = self._grad_h or self._output.new() -- size: (T, N, R)
   
   if not c0 then c0 = self.c0 end
   if not h0 then h0 = self.h0 end

   local grad_x, grad_c, grad_h = self._grad_x, self._grad_c, self._grad_h
   local h, c = self._output, self.cell
   
   local Wx = self.weight:narrow(1,1,D):narrow(2, 1, 3 * H)
   local Wh = self.weight:narrow(1,D+1,R):narrow(2, 1, 3 * H)
   local Wfx = self.weight:narrow(1,1,D):narrow(2, 3 * H + 1, H)
   local Wfh = self.weight:narrow(1,D+1,R):narrow(2, 3 * H + 1, H)
   local grad_Wx = self.gradWeight:narrow(1,1,D):narrow(2, 1, 3 * H)
   local grad_Wh = self.gradWeight:narrow(1,D+1,R):narrow(2, 1, 3 * H)
   local grad_Wfx = self.gradWeight:narrow(1,1,D):narrow(2, 3 * H + 1, H)
   local grad_Wfh = self.gradWeight:narrow(1,D+1,R):narrow(2, 3 * H + 1, H)
   local grad_b = self.gradBias:narrow(1, 1, 3 * H)
   local grad_fb = self.gradBias:narrow(1, 3 * H + 1, H)

   -- grad_h0:resizeAs(h0):zero()
   -- grad_c0:resizeAs(c0):zero()
   grad_x:resizeAs(x):zero()
   grad_c:resizeAs(c):zero()
   grad_h:resizeAs(h):zero()
   self.buffer0:resize(N, T, H):zero() -- size: (N, T, H)
   self.buffer1:resize(N, H):zero() -- size: (N, H)
   self.buffer2:resize(N, H):zero() -- size: (N, H)
   -- self.grad_next_h = self.gradPrevOutput and self.buffer1:copy(self.gradPrevOutput) or self.buffer1:zero()
   -- local grad_next_c = self.userNextGradCell and self.buffer2:copy(self.userNextGradCell) or self.buffer2:zero()
   
   for t = T, 1, -1 do
      local next_h, next_c, grad_next_c, grad_next_h = h[t], c[t], grad_c[t], grad_h[t]
      grad_next_h:add(grad_out_h[t])
      
      if self.maskzero and torch.type(self) ~= 'nn.ChildSumLSTM' then 
         -- we only do this for sub-classes (LSTM doesn't need it)   
         -- build mask from input
         local cur_x = x[t]
         local vectorDim = cur_x:dim()
         self._zeroMask = self._zeroMask or cur_x.new()
         self._zeroMask:norm(cur_x, 2, vectorDim)
         self.zeroMask = self.zeroMask or ((torch.type(cur_x) == 'torch.CudaTensor') and torch.CudaByteTensor() or torch.ByteTensor())
         self._zeroMask.eq(self.zeroMask, self._zeroMask, 0)
         -- zero masked gradOutput
         self:recursiveMask(self.grad_next_h, self.zeroMask)
      end
      
      -- for LSTMP
      self:gradAdapter(scale, t)

      local i = self.gates[{t, {}, {1, H}}]
      local o = self.gates[{t, {}, {H + 1, 2 * H}}]
      local g = self.gates[{t, {}, {2 * H + 1, 3 * H}}]
      local f = self.fgates[t] --  size: (N, T, H)
   
      local grad_a = self.grad_a_buffer:resize(N, 3 * H):zero()
      local grad_f = self.grad_f_buffer:resize(N, T, H):zero() -- grads for fgates size: (N, T, H)
      local grad_ai = grad_a[{{}, {1, H}}]   -- size: (N, H)
      local grad_ao = grad_a[{{}, {H + 1, 2 * H}}]    -- size: (N, H)
      local grad_ag = grad_a[{{}, {2 * H + 1, 3 * H}}]   -- size: (N, H)
      
      -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
      -- to compute grad_next_c. We will need tanh_next_c (stored in grad_ai)
      -- to compute grad_ao; the other values can be overwritten after we compute
      -- grad_next_c
      local tanh_next_c = grad_ai:tanh(next_c) -- size: (N, H)
      local tanh_next_c2 = grad_ag:cmul(tanh_next_c, tanh_next_c) -- size: (N, H)
      local my_grad_next_c = grad_ao -- size: (N, H)
      my_grad_next_c:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_next_h)
      grad_next_c:add(my_grad_next_c) -- size: (N, H)
      
      -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after
      -- that we can overwrite it.
      grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_next_h)

      -- Use grad_ai as a temporary buffer for computing grad_ag
      local g2 = grad_ai:cmul(g, g)
      grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_next_c)

      -- We don't need any temporary storage for these so do them last
      grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_next_c)
      -- grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_next_c)
      
      self.buffer0:cmul(h:transpose(1, 2), connect_expand[t]) -- size: (N, T, H)
      
      grad_x[t]:mm(grad_a, Wx:t()) -- backprop to input from i,o,g
      grad_Wx:addmm(scale, x[t]:t(), grad_a)
      grad_Wh:addmm(scale, self.buffer0:sum(2):view(N, H):t(), grad_a)
      self.buffer2:mm(grad_a, Wh:t()) -- size: (N, H)
      self.buffer0:cmul(self.buffer2:view(N, 1, H):expand(N, T, H), connect_expand[t]) -- size: (N, T, H)
      grad_h:add(self.buffer0:transpose(1, 2)) -- backprop to predecessors' output from i,o,g
      -- grad_Wh:addmm(scale, prev_h:t(), grad_a)
      
      self.buffer0:cmul(c:transpose(1, 2), connect_expand[t]) -- size: (N, T, H)
      grad_f:fill(1):add(-1, f):cmul(f):cmul(self.buffer0):cmul(grad_next_c:view(N, 1, H):expand(N, T, H)) -- size: (N, T, H)
      grad_c:transpose(1, 2):addcmul(grad_next_c:view(N, 1, H):expand(N, T, H), torch.cmul(f, connect_expand[t])) -- backprop to predecessors' cells
      grad_x[t]:addmm(grad_f:sum(2):view(N, H), Wfx:t())
      grad_Wfx:addmm(scale, x[t]:t(), grad_f:sum(2):view(N, H))
      
      self.buffer0:zero():addcmul(h:transpose(1, 2), connect_expand[t]) -- size: (N, T, H)
      for j = 1, N do
         grad_Wfh:addmm(scale, self.buffer0[j]:t(), grad_f[j])
         grad_h:select(2, j):addmm(grad_f[j], Wfh:t()) -- backprop to predecessors' output from f
      end

      local grad_a_sum = self.buffer3:resize(1, 4 * H):narrow(2, 1, 3 * H):sum(grad_a, 1)
      grad_b:add(scale, grad_a_sum)
      local grad_f_sum = self.buffer3:narrow(2, 3 * H + 1, H):copy(grad_f:sum(2):sum(1):view(1, H))
      grad_fb:add(scale, grad_f_sum)
      
   end
   
   if self.batchfirst then
      self.grad_x = grad_x:transpose(1,2) -- T x N -> N x T
   else
      self.grad_x = grad_x
   end
   
   if self._return_grad_c0 and self._return_grad_h0 then
      self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
   elseif self._return_grad_h0 then
      self.gradInput = {self.grad_h0, self.grad_x}
   else
      self.gradInput = self.grad_x
   end

   return self.gradInput
end

function ChildSumLSTM:gradAdapter(scale, t)
   -- Placeholder for SeqLSTMP
end

function ChildSumLSTM:clearState()
   self.cell:set()
   self.gates:set()
   self.fgates:set()
   self.buffer0:set()
   self.buffer1:set()
   self.buffer2:set()
   self.buffer3:set()
   self.grad_a_buffer:set()
   self.grad_f_buffer:set()

   self.grad_c0:set()
   self.grad_h0:set()
   self.grad_x:set()
   self.grad_h = nil
   self.grad_c = nil
   self.output:set()
   self._output = nil
   self.gradInput = nil
   
   self.zeroMask = nil
   self._zeroMask = nil
   self._maskbyte = nil
   self._maskindices = nil
end

function ChildSumLSTM:updateGradInput(input, gradOutput)
   if self.recompute_backward then
      self:backward(input, gradOutput, 1.0)
   end
   return self.gradInput
end

function ChildSumLSTM:accGradParameters(input, gradOutput, scale)
   if self.recompute_backward then
      self:backward(input, gradOutput, scale)
   end
end

function ChildSumLSTM:forget()
   self.c0:resize(0)
   self.h0:resize(0)
end

function ChildSumLSTM:type(type, ...)
   self.zeroMask = nil
   self._zeroMask = nil
   self._maskbyte = nil
   self._maskindices = nil
   return parent.type(self, type, ...)
end

-- Toggle to feed long sequences using multiple forwards.
-- 'eval' only affects evaluation (recommended for RNNs)
-- 'train' only affects training
-- 'neither' affects neither training nor evaluation
-- 'both' affects both training and evaluation (recommended for LSTMs)
ChildSumLSTM.remember = nn.Sequencer.remember

function ChildSumLSTM:training()
   if self.train == false then
      -- forget at the start of each training
      self:forget()
   end
   parent.training(self)
end

function ChildSumLSTM:evaluate()
   if self.train ~= false then
      -- forget at the start of each evaluation
      self:forget()
   end
   parent.evaluate(self)
   assert(self.train == false)
end

function ChildSumLSTM:maskZero()
   self.maskzero = true
end
