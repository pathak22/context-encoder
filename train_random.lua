require 'torch'
require 'nn'
require 'optim'
require 'image'
util = paths.dofile('util.lua')

opt = {
   batchSize = 64,         -- number of samples to produce
   loadSize = 350,         -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
   fineSize = 128,         -- size of random crops. Only 64 and 128 supported.
   nBottleneck = 100,      -- #  of dim for bottleneck of encoder
   nef = 64,               -- #  of encoder filters in first conv layer
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nc = 3,                 -- # of channels in input
   wtl2 = 0,               -- 0 means don't use else use with this weight
   useOverlapPred = 0,        -- overlapping edges (1 means yes, 0 means no). 1 means put 10x more L2 weight on unmasked region.
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_iter = 50,      -- # number of iterations after which display is updated
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'train1',        -- name of the experiment you are running
   manualSeed = 0,         -- 0 means random seed

   -- Extra Options:
   conditionAdv = 0,       -- 0 means false else true
   noiseGen = 0,           -- 0 means false else true
   noisetype = 'normal',   -- uniform / normal
   nz = 100,               -- #  of dim for Z
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
if opt.conditionAdv == 0 then opt.conditionAdv = false end
if opt.noiseGen == 0 then opt.noiseGen = false end

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())

---------------------------------------------------------------------------
-- Initialize network variables
---------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = opt.nc
local nz = opt.nz
local nBottleneck = opt.nBottleneck
local ndf = opt.ndf
local ngf = opt.ngf
local nef = opt.nef
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

---------------------------------------------------------------------------
-- Generator net
---------------------------------------------------------------------------
-- Encode Input Context to noise (architecture similar to Discriminator)
local netE = nn.Sequential()
-- input is (nc) x 128 x 128
netE:add(SpatialConvolution(nc, nef, 4, 4, 2, 2, 1, 1))
netE:add(nn.LeakyReLU(0.2, true))
if opt.fineSize == 128 then
  -- state size: (nef) x 64 x 64
  netE:add(SpatialConvolution(nef, nef, 4, 4, 2, 2, 1, 1))
  netE:add(SpatialBatchNormalization(nef)):add(nn.LeakyReLU(0.2, true))
end
-- state size: (nef) x 32 x 32
netE:add(SpatialConvolution(nef, nef * 2, 4, 4, 2, 2, 1, 1))
netE:add(SpatialBatchNormalization(nef * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (nef*2) x 16 x 16
netE:add(SpatialConvolution(nef * 2, nef * 4, 4, 4, 2, 2, 1, 1))
netE:add(SpatialBatchNormalization(nef * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (nef*4) x 8 x 8
netE:add(SpatialConvolution(nef * 4, nef * 8, 4, 4, 2, 2, 1, 1))
netE:add(SpatialBatchNormalization(nef * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (nef*8) x 4 x 4
netE:add(SpatialConvolution(nef * 8, nBottleneck, 4, 4))
-- state size: (nBottleneck) x 1 x 1

local netG = nn.Sequential()
local nz_size = nBottleneck
if opt.noiseGen then
    local netG_noise = nn.Sequential()
    -- input is Z: (nz) x 1 x 1, going into a convolution
    netG_noise:add(SpatialConvolution(nz, nz, 1, 1, 1, 1, 0, 0))
    -- state size: (nz) x 1 x 1

    local netG_pl = nn.ParallelTable();
    netG_pl:add(netE)
    netG_pl:add(netG_noise)

    netG:add(netG_pl)
    netG:add(nn.JoinTable(2))
    netG:add(SpatialBatchNormalization(nBottleneck+nz)):add(nn.LeakyReLU(0.2, true))
    -- state size: (nBottleneck+nz) x 1 x 1

    nz_size = nBottleneck+nz
else
    netG:add(netE)
    netG:add(SpatialBatchNormalization(nBottleneck)):add(nn.LeakyReLU(0.2, true))

    nz_size = nBottleneck
end

-- Decode noise to generate image
-- input is Z: (nz_size) x 1 x 1, going into a convolution
netG:add(SpatialFullConvolution(nz_size, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
if opt.fineSize == 128 then
  netG:add(SpatialFullConvolution(ngf, ngf, 4, 4, 2, 2, 1, 1))
  netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
  -- state size: (ngf) x 64 x 64
end
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 128 x 128

netG:apply(weights_init)

---------------------------------------------------------------------------
-- Adversarial discriminator net
---------------------------------------------------------------------------
local netD = nn.Sequential()
if opt.conditionAdv then
    print('conditional adv not implemented')
    exit()
    local netD_ctx = nn.Sequential()
    -- input Context: (nc) x 128 x 128, going into a convolution
    netD_ctx:add(SpatialConvolution(nc, ndf, 5, 5, 2, 2, 2, 2))
    -- state size: (ndf) x 64 x 64

    local netD_pred = nn.Sequential()
    -- input pred: (nc) x 64 x 64, going into a convolution
    netD_pred:add(SpatialConvolution(nc, ndf, 5, 5, 2, 2, 2+32, 2+32))      -- 32: to keep scaling of features same as context
    -- state size: (ndf) x 64 x 64

    local netD_pl = nn.ParallelTable();
    netD_pl:add(netD_ctx)
    netD_pl:add(netD_pred)

    netD:add(netD_pl)
    netD:add(nn.JoinTable(2))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf * 2) x 64 x 64

    netD:add(SpatialConvolution(ndf*2, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 32 x 32
else
    -- input is (nc) x 128 x 128, going into a convolution
    netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 64 x 64
end
if opt.fineSize == 128 then
  netD:add(SpatialConvolution(ndf, ndf, 4, 4, 2, 2, 1, 1))
  netD:add(SpatialBatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
  -- state size: (ndf) x 32 x 32
end
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)

---------------------------------------------------------------------------
-- Loss Metrics
---------------------------------------------------------------------------
local criterion = nn.BCECriterion()
local criterionMSE
if opt.wtl2~=0 then
  criterionMSE = nn.MSECriterion()
end

---------------------------------------------------------------------------
-- Setup Solver
---------------------------------------------------------------------------
print('LR of Gen is ',(opt.wtl2>0 and opt.wtl2<1) and 10 or 1,'times Adv')
optimStateG = {
   learningRate = (opt.wtl2>0 and opt.wtl2<1) and opt.lr*10 or opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

---------------------------------------------------------------------------
-- Initialize data variables
---------------------------------------------------------------------------
local mask_global = torch.ByteTensor(opt.batchSize, opt.fineSize, opt.fineSize)
local input_ctx_vis = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local input_ctx = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local input_center = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local input_real_center
if opt.wtl2~=0 then
    input_real_center = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
end
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG, errG_l2
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

if pcall(require, 'cudnn') and pcall(require, 'cunn') and opt.gpu>0 then
    print('Using CUDNN !')
end
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input_ctx_vis = input_ctx_vis:cuda(); input_ctx = input_ctx:cuda();  input_center = input_center:cuda()
   noise = noise:cuda();  label = label:cuda()
   netG = util.cudnn(netG);     netD = util.cudnn(netD)
   netD:cuda();           netG:cuda();           criterion:cuda();      
   if opt.wtl2~=0 then
      criterionMSE:cuda(); input_real_center = input_real_center:cuda();
   end
end
print('NetG:',netG)
print('NetD:',netD)

-- Generating random pattern
local res = 0.06 -- the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
local density = 0.25
local MAX_SIZE = 10000
local low_pattern = torch.Tensor(res*MAX_SIZE, res*MAX_SIZE):uniform(0,1):mul(255)
local pattern = image.scale(low_pattern, MAX_SIZE, MAX_SIZE,'bicubic')
low_pattern = nil
pattern:div(255);
pattern = torch.lt(pattern,density):byte()  -- 25% 1s and 75% 0s
pattern = pattern:byte()
print('...Random pattern generated')

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noisetype == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise_vis:normal(0, 1)
end

---------------------------------------------------------------------------
-- Define generator and adversary closures
---------------------------------------------------------------------------
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real_ctx = data:getBatch()
   real_center = real_ctx  -- view
   input_center:copy(real_center)
   if opt.wtl2~=0 then
      input_real_center:copy(real_center)
   end

   -- get random mask
   local mask, wastedIter
   wastedIter = 0
   while true do
     local x = torch.uniform(1, MAX_SIZE-opt.fineSize)
     local y = torch.uniform(1, MAX_SIZE-opt.fineSize)
     mask = pattern[{{y,y+opt.fineSize-1},{x,x+opt.fineSize-1}}]  -- view, no allocation
     local area = mask:sum()*100./(opt.fineSize*opt.fineSize)
     if area>20 and area<30 then  -- want it to be approx 75% 0s and 25% 1s
        -- print('wasted tries: ',wastedIter)
        break
     end
     wastedIter = wastedIter + 1
   end
   torch.repeatTensor(mask_global,mask,opt.batchSize,1,1)

   real_ctx[{{},{1},{},{}}][mask_global] = 2*117.0/255.0 - 1.0
   real_ctx[{{},{2},{},{}}][mask_global] = 2*104.0/255.0 - 1.0
   real_ctx[{{},{3},{},{}}][mask_global] = 2*123.0/255.0 - 1.0
   input_ctx:copy(real_ctx)
   data_tm:stop()

   label:fill(real_label)
   local output
   if opt.conditionAdv then
      output = netD:forward({input_ctx,input_center})
   else
      output = netD:forward(input_center)
   end
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   if opt.conditionAdv then
      netD:backward({input_ctx,input_center}, df_do)
   else
      netD:backward(input_center, df_do)
   end
   
   -- train with fake
   if opt.noisetype == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noisetype == 'normal' then
       noise:normal(0, 1)
   end
   local fake
   if opt.noiseGen then
      fake = netG:forward({input_ctx,noise})
   else
      fake = netG:forward(input_ctx)
   end
   input_center:copy(fake)
   label:fill(fake_label)

   local output
   if opt.conditionAdv then
      output = netD:forward({input_ctx,input_center})
   else
      output = netD:forward(input_center)
   end
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   if opt.conditionAdv then
      netD:backward({input_ctx,input_center}, df_do)
   else
      netD:backward(input_center, df_do)
   end

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward({input_ctx,noise})
   input_center:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward({input_ctx,input_center}) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg
   if opt.conditionAdv then
      df_dg = netD:updateGradInput({input_ctx,input_center}, df_do)
      df_dg = df_dg[2]     -- df_dg[2] because conditional GAN
   else
      df_dg = netD:updateGradInput(input_center, df_do)
   end

   local errG_total = errG
   if opt.wtl2~=0 then
      errG_l2 = criterionMSE:forward(input_center, input_real_center)
      local df_dg_l2 = criterionMSE:backward(input_center, input_real_center)

      if opt.useOverlapPred==0 then
        if (opt.wtl2>0 and opt.wtl2<1) then
          df_dg:mul(1-opt.wtl2):add(opt.wtl2,df_dg_l2)
          errG_total = (1-opt.wtl2)*errG + opt.wtl2*errG_l2
        else
          df_dg:add(opt.wtl2,df_dg_l2)
          errG_total = errG + opt.wtl2*errG_l2
        end
      else
        local overlapL2Weight = 10
        local wtl2Matrix = df_dg_l2:clone():fill(overlapL2Weight*opt.wtl2)
        for i=1,3 do
          wtl2Matrix[{{},{i},{},{}}][mask_global] = opt.wtl2
        end
        if (opt.wtl2>0 and opt.wtl2<1) then
          df_dg:mul(1-opt.wtl2):addcmul(1,wtl2Matrix,df_dg_l2)
          errG_total = (1-opt.wtl2)*errG + opt.wtl2*errG_l2
        else
          df_dg:addcmul(1,wtl2Matrix,df_dg_l2)
          errG_total = errG + opt.wtl2*errG_l2
        end
      end
   end

   if opt.noiseGen then
      netG:backward({input_ctx,noise}, df_dg)
   else
      netG:backward(input_ctx, df_dg)
   end

   return errG_total, gradParametersG
end

---------------------------------------------------------------------------
-- Train Context Encoder
---------------------------------------------------------------------------
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % opt.display_iter == 0 and opt.display then
          local real_ctx = data:getBatch()
          -- disp.image(real_ctx, {win=opt.display_id * 6, title=opt.name})

          local mask, wastedIter
          wastedIter = 0
          while true do
           local x = torch.uniform(1, MAX_SIZE-opt.fineSize)
           local y = torch.uniform(1, MAX_SIZE-opt.fineSize)
           mask = pattern[{{y,y+opt.fineSize-1},{x,x+opt.fineSize-1}}]  -- view, no allocation
           local area = mask:sum()*100./(opt.fineSize*opt.fineSize)
           if area>20 and area<30 then  -- want it to be approx 75% 0s and 25% 1s
              -- print('wasted tries: ',wastedIter)
              break
           end
           wastedIter = wastedIter + 1
          end
          mask=torch.repeatTensor(mask,opt.batchSize,1,1)

          real_ctx[{{},{1},{},{}}][mask] = 2*117.0/255.0 - 1.0
          real_ctx[{{},{2},{},{}}][mask] = 2*104.0/255.0 - 1.0
          real_ctx[{{},{3},{},{}}][mask] = 2*123.0/255.0 - 1.0
          input_ctx_vis:copy(real_ctx)

          local fake
          if opt.noiseGen then
            fake = netG:forward({input_ctx_vis,noise_vis})
          else
            fake = netG:forward(input_ctx_vis)
          end
          disp.image(fake, {win=opt.display_id, title=opt.name})
          
          real_ctx[{{},{1},{},{}}][mask] = 1.0
          real_ctx[{{},{2},{},{}}][mask] = 1.0
          real_ctx[{{},{3},{},{}}][mask] = 1.0
          disp.image(real_ctx, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G_L2: %.4f   Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real, errG_l2 or -1,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   if epoch % 20 == 0 then
      util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG, opt.gpu)
      util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD, opt.gpu)
   end
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
