require 'image'
require 'nn'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 21,        -- number of samples to produce
    net = '',              -- path to the generator network
    imDir = '',            -- directory containing pred_center 
    name = 'demo_out',     -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
    nc = 3,                -- # of channels in input
    manualSeed = 0,        -- 0 means random seed
    overlapPred = 0,       -- overlapping edges of center with context
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- load Context-Encoder
assert(opt.net ~= '', 'provide a generator model')
net = torch.load(opt.net)
net:apply(function(m) if m.weight then 
    m.gradWeight = m.weight:clone():zero(); 
    m.gradBias = m.bias:clone():zero(); end end)
net:evaluate()

-- initialize variables
inputSize = 128
image_ctx = torch.Tensor(opt.batchSize, opt.nc, inputSize, inputSize)
input_image_ctx = torch.Tensor(opt.batchSize, opt.nc, inputSize, inputSize)

-- port to GPU
if opt.gpu > 0 then
    require 'cunn'
    net:cuda()
    input_image_ctx = input_image_ctx:cuda()
else
   net:float()
end
print(net)

-- load data
for i=1,opt.batchSize do
    local imPath = string.format(opt.imDir.."/%03d_im.png",i)
    local input = image.load(imPath, nc, 'float')
    input = image.scale(input, inputSize, inputSize)
    input:mul(2):add(-1)
    image_ctx[i]:copy(input)
end
print('Loaded Image Block: ', image_ctx:size(1)..' x '..image_ctx:size(2) ..' x '..image_ctx:size(3)..' x '..image_ctx:size(4))

-- remove center region from input image
real_center = image_ctx[{{},{},{1 + inputSize/4, inputSize/2 + inputSize/4},{1 + inputSize/4, inputSize/2 + inputSize/4}}]:clone()      -- copy by value

-- fill center region with mean value
image_ctx[{{},{1},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}] = 2*117.0/255.0 - 1.0
image_ctx[{{},{2},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}] = 2*104.0/255.0 - 1.0
image_ctx[{{},{3},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}] = 2*123.0/255.0 - 1.0
input_image_ctx:copy(image_ctx)

-- run Context-Encoder to inpaint center
pred_center = net:forward(input_image_ctx)
print('Prediction: size: ', pred_center:size(1)..' x '..pred_center:size(2) ..' x '..pred_center:size(3)..' x '..pred_center:size(4))
print('Prediction: Min, Max, Mean, Stdv: ', pred_center:min(), pred_center:max(), pred_center:mean(), pred_center:std())

-- paste predicted center in the context
image_ctx[{{},{},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}]:copy(pred_center[{{},{},{1 + opt.overlapPred, inputSize/2 - opt.overlapPred},{1 + opt.overlapPred, inputSize/2 - opt.overlapPred}}])

-- re-transform scale back to normal
input_image_ctx:add(1):mul(0.5)
image_ctx:add(1):mul(0.5)
pred_center:add(1):mul(0.5)
real_center:add(1):mul(0.5)

-- save outputs
-- image.save(opt.name .. '_predWithContext.png', image.toDisplayTensor(image_ctx))
-- image.save(opt.name .. '_realCenter.png', image.toDisplayTensor(real_center))
-- image.save(opt.name .. '_predCenter.png', image.toDisplayTensor(pred_center))

-- save outputs in a pretty manner
real_center=nil; pred_center=nil;
pretty_output = torch.Tensor(2*opt.batchSize, opt.nc, inputSize, inputSize)
input_image_ctx[{{},{1},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}] = 1
input_image_ctx[{{},{2},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}] = 1
input_image_ctx[{{},{3},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred},{1 + inputSize/4 + opt.overlapPred, inputSize/2 + inputSize/4 - opt.overlapPred}}] = 1
for i=1,opt.batchSize do
    pretty_output[2*i-1]:copy(input_image_ctx[i])
    pretty_output[2*i]:copy(image_ctx[i])
end
image.save(opt.name .. '.png', image.toDisplayTensor(pretty_output))
print('Saved predictions to: ./', opt.name .. '.png')
