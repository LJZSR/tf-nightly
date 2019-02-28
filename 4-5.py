from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
savedir = 'log/'
print_tensors_in_checkpoint_file(savedir+'linermodel.cpkt', None, True)
