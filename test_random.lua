require 'image'
require 'nn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 30,        -- number of samples to produce
    net = '',              -- path to the generator network
    name = 'test1',        -- name of the experiment and prefix of file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
    nc = 3,                -- # of channels in input
    display = 1,           -- Display image: 0 = false, 1 = true
    loadSize = 0,          -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
    fineSize = 128,        -- size of random crops
    nThreads = 1,          -- # of data loading threads to use
    manualSeed = 0,        -- 0 means random seed
    useOverlapPred = 0,        -- overlapping edges (1 means yes, 0 means no). 1 means put 10x more L2 weight on unmasked region.

    -- Extra Options:
    noiseGen = 0,          -- 0 means false else true; only works if network was trained with noise too.
    noisetype = 'normal',  -- type of noise distribution (uniform / normal)
    nz = 100,              -- length of noise vector if used
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
if opt.noiseGen == 0 then opt.noiseGen = false end

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- load Context-Encoder
assert(opt.net ~= '', 'provide a generator model')
net = util.load(opt.net, opt.gpu)
net:evaluate()

-- initialize variables
input_image_ctx = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
local noise 
if opt.noiseGen then
    noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
    if opt.noisetype == 'uniform' then
        noise:uniform(-1, 1)
    elseif opt.noisetype == 'normal' then
        noise:normal(0, 1)
    end
end

-- port to GPU
if opt.gpu > 0 then
    require 'cunn'
    if pcall(require, 'cudnn') then
        print('Using CUDNN !')
        require 'cudnn'
        net = util.cudnn(net)
    end
    net:cuda()
    input_image_ctx = input_image_ctx:cuda()
    if opt.noiseGen then
        noise = noise:cuda()
    end
else
   net:float()
end
print(net)

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

-- load data
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
local image_ctx = data:getBatch()
print('Loaded Image Block: ', image_ctx:size(1)..' x '..image_ctx:size(2) ..' x '..image_ctx:size(3)..' x '..image_ctx:size(4))

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
mask=torch.repeatTensor(mask,opt.batchSize,1,1)

-- original input image
real_center = image_ctx:clone() -- copy by value

-- fill masked region with mean value
image_ctx[{{},{1},{},{}}][mask] = 2*117.0/255.0 - 1.0
image_ctx[{{},{2},{},{}}][mask] = 2*104.0/255.0 - 1.0
image_ctx[{{},{3},{},{}}][mask] = 2*123.0/255.0 - 1.0
input_image_ctx:copy(image_ctx)

-- run Context-Encoder to inpaint center
local pred_center
if opt.noiseGen then
    pred_center = net:forward({input_image_ctx,noise})
else
    pred_center = net:forward(input_image_ctx)
end
print('Prediction: size: ', pred_center:size(1)..' x '..pred_center:size(2) ..' x '..pred_center:size(3)..' x '..pred_center:size(4))
print('Prediction: Min, Max, Mean, Stdv: ', pred_center:min(), pred_center:max(), pred_center:mean(), pred_center:std())

-- paste predicted region in the context
image_ctx[{{},{1},{},{}}][mask] = pred_center[{{},{1},{},{}}][mask]:float()
image_ctx[{{},{2},{},{}}][mask] = pred_center[{{},{2},{},{}}][mask]:float()
image_ctx[{{},{3},{},{}}][mask] = pred_center[{{},{3},{},{}}][mask]:float()

-- re-transform scale back to normal
input_image_ctx:add(1):mul(0.5)
image_ctx:add(1):mul(0.5)
pred_center:add(1):mul(0.5)
real_center:add(1):mul(0.5)

-- save outputs
-- image.save(opt.name .. '_predWithContext.png', image.toDisplayTensor(image_ctx))
-- image.save(opt.name .. '_realCenter.png', image.toDisplayTensor(real_center))
-- image.save(opt.name .. '_predCenter.png', image.toDisplayTensor(pred_center))

if opt.display then
    disp = require 'display'
    disp.image(pred_center, {win=1000, title=opt.name})
    -- disp.image(real_center, {win=1001, title=opt.name})
    disp.image(image_ctx, {win=1002, title=opt.name})
    print('Displayed image in browser !')
end

-- save outputs in a pretty manner
real_center=nil; pred_center=nil;
pretty_output = torch.Tensor(2*opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
input_image_ctx[{{},{1},{},{}}][mask] = 1.0
input_image_ctx[{{},{2},{},{}}][mask] = 1.0
input_image_ctx[{{},{3},{},{}}][mask] = 1.0
for i=1,opt.batchSize do
    pretty_output[2*i-1]:copy(input_image_ctx[i])
    pretty_output[2*i]:copy(image_ctx[i])
end
image.save(opt.name .. '.png', image.toDisplayTensor(pretty_output))
print('Saved predictions to: ./', opt.name .. '.png')
