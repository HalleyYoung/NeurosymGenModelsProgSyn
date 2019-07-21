# NeurosymGenModelsProgSyn
the code for the ICML paper (Learning Neurosymbolic Generative Models via Program Synthesis)
\]

# To run this code:
Clone this directory.  Make a directory called ``facades``, make a directory inside their called data, and download the facades dataset to that directory.

# To download the data for the facades dataset:
Go to http://cmp.felk.cvut.cz/~tylecr1/facade/, extract the facade datasets and select only the .jpgs.  Also download from the other links on that site.

# To create a test-train split, run createtesttrain.py.  This will create a test/ folder and train/ folder in the facades directory.

# To extract the full programs from all the files in the ``train`` dataset, first run ``gendiffmatstrainfull.py``, and then run ``genprogtrainfull.py.``

# To extract the partially observed programs from all the files in the ``train`` dataset, first run ``gendiffmatstrainthird.py``, and then run ``genprogtrainthird.py``.

# To learn a model from the partially observed programs to the full programs, run ``genprogtoprogtrain.py``.

# To render a partial extrapolated image using the generated generative model, run ``renderpartialprogtrain.py`` . This will generate a new folder in the ``facades`` folder called ``facadescycledata`` with two subdirectories: one called "trainA", and another called "trainB".

# To extract the partially observed programs from all the files in the ``test`` dataset, first run ``gendiffmatstestthird.py``, and then run ``genprogtestthird.py``.

# To use the learned model to render a partial extrapolated image for the test files, run ``renderpartialprogtest.py``.  This will create a new folder in the ``facadescycledata`` folder called ``testA``.  

# Clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix, and cd into it.  Copy the folder ``facades/facadescycledata`` into pytorch-CycleGAN-and-pix2pix/datasets.

# In the pytorch-CycleGAN-and-pix2pix folder, create the CycleGAN model using the command `python train.py --dataroot ./datasets/facadescycledata --name completion_cyclegan --model cycle_gan --display_id 0 --batch_size 2 --gpu_ids 0 --input_nc 3 --output_nc 3 --direction AtoB`

# Generate the final completions of the test data by, in the pytorch-CycleGAN-and-pix2pix folder, running `python test.py --dataroot ./datasets/facadescycledata --name completion_cyclegan --model cycle_gan --input_nc 3 --output_nc 3`
