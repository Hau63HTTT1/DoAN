from torch.utils.data import Dataset
import random


class BaseDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.is_train = self.phase == 'train'
        self.projection_mode = 'orthogonal' 
    def __len__(self):
        return 0

    def get_item(self, index):
     
        try:
            res = {
                'name': None,  
                'b_min': None,  
                'b_max': None,  

                'samples': None,  
                'labels': None,  

                'img': None,  
                'calib': None,  
                'extrinsic': None,  
                'mask': None,  
            }
            return res
        except:
            print("Requested index %s has missing files. Using a random sample instead." % index)
            return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)
