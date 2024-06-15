import os

sample = True

basic_data_dir = '/Users/hogan/Desktop/AICode/traindata'



def data_dir_with(catergory):
    local_dir = basic_data_dir
    if sample:
        local_dir = os.path.join(local_dir, 'sampledata')
    else:
        local_dir = os.path.join(local_dir, 'truedata')
    re = os.path.join(local_dir,catergory)
    print(f'using training data:{re}')
    return re