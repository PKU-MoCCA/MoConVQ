from motion_utils import *
from misc import add_to_list
import pytorch_utils as ptu
import traceback
import VclSimuBackend
try:
    from VclSimuBackend.ODESim.BVHToTarget import BVHToTargetBase
    # from VclSimuBackend.ODESim.TargetPose import TargetPose
    # from VclSimuBackend.ODESim.ODECharacter import ODECharacter
    from VclSimuBackend.ODESim.TargetPose import SetTargetToCharacter
except ImportError:
    BVHToTargetBase = VclSimuBackend.ODESim.BVHToTarget.BVHToTargetBase
    SetTargetToCharacter = VclSimuBackend.ODESim.TargetPose.SetTargetToCharacter


class MotionDataSet():
    def __init__(self, fps) -> None:
        """ We orgnaize motion capture data into pickle files

        Args:
            fps (int): target fps of downsampled bvh
        """        
        # init buffer
        self.state = None
        self.observation = None
        self.done = None
        self.future = None
        
        # init fps
        self.fps = fps
    
    @property
    def stastics(self):
        obs_mean = np.mean(self.observation, axis=0)
        obs_std = np.std(self.observation, axis =0)
        obs_std[obs_std < 1e-1] = 0.1
        
        delta = self.observation[1:] - self.observation[:-1]
        _,_,vel, avel,_,_= decompose_obs(delta)
        num = delta.shape[0]
        delta = np.concatenate([vel.reshape(num,-1,3),avel.reshape(num,-1,3)], axis = -1)
        delta = delta.reshape(num,-1)
        delta_mean = np.mean(delta, axis = 0)
        delta_std = np.std(delta, axis = 0)
        delta_std[delta_std < 1e-1] = 0.1
        future_mean = np.mean(self.future, axis = 0)
        future_std = np.std(self.future, axis = 0)
        return {
            'obs_mean': obs_mean,
            'obs_std': obs_std,
            'delta_mean': delta_mean,
            'delta_std': delta_std,
            'future_mean': future_mean,
            'future_std': future_std
        }
    
    def add_bvh_with_character(self, name, character, flip = False):
        if flip:
            bvh_target_base = BVHToTargetBase(name, self.fps, character, flip = np.array([1,0,0]))
        else:
            bvh_target_base = BVHToTargetBase(name, self.fps, character)
            
        target = bvh_target_base.init_target()        
        target_setter : SetTargetToCharacter = SetTargetToCharacter(character, target)

        state, ob, done = [],[],[] 
        
        offset = np.zeros(3)
        for i in range(10):
            target_setter.set_character_byframe(i)
            aabb = character.get_aabb()
            offset += np.array([0,-aabb[2]+1e-3,0])
        offset /= 10
        # offset[1] -= 0.05
        
        for i in range(target.num_frames):
            target_setter.set_character_byframe(i)
            character.move_character_by_delta(offset)
            state_tmp = character_state(character)
            ob_tmp =  state2ob(torch.from_numpy(state_tmp)).numpy()
            done_tmp = (i == (target.num_frames -1))
            state.append(state_tmp[None,...])
            ob.append(ob_tmp.flatten()[None,...])
            done.append(np.array(done_tmp).reshape(1,1))

        state = np.concatenate(state, axis = 0)
        future = states2future(state)
        
        self.state = add_to_list(state, self.state)
        self.observation = add_to_list(ob, self.observation)
        self.done = add_to_list(done, self.done)
        self.future = add_to_list(future, self.future)
        
        return target, target_setter, offset
         
    def add_folder_bvh(self, name, character, mirror_augment = True):
        """Add every bvh in a forlder into motion dataset

        Args:
            name (str): path of bvh folder
            character (ODECharacter): the character of ode
            mirror_augment (bool, optional): whether to use mirror augment. Defaults to True.
        """                
        target_setters = []
        for file in os.listdir(name):
            if '.bvh' in file:
                print(f'add {file}')
                target_setter = self.add_bvh_with_character(os.path.join(name, file), character)
                target_setters.append(target_setter)
        if mirror_augment:
            for file in os.listdir(name):
                if '.bvh' in file:
                    print(f'add {file} flip')
                    target_setter = self.add_bvh_with_character(os.path.join(name, file), character, flip = True)
                    target_setters.append(target_setter)
                    
        return target_setters

import h5py
class HDF5MotionDataset():
    '''
    usage:
        dataset = HDF5MotionDataset(fps)
        dataset.begin_create('test.hdf5')
        dataset.add_folder_bvh('bvh', character)
        dataset.end_create()
        
        # loading:
        dataset = HDF5MotionDataset(fps)
        dataset.load('test.hdf5')
    '''
    def __init__(self, fps) -> None:
        self.motions_file = None
        self.fps = fps
    
    def begin_create(self, name, clip_feature_set = None):
        self.motions_file = h5py.File(name, 'w')
        self.clip_feature_set = None #h5py.File(clip_feature_set, 'r')
    
    def continue_create(self, name, clip_feature_set = None):
        self.motions_file = h5py.File(name, 'a')
        self.clip_feature_set = h5py.File(clip_feature_set, 'r')
     
    def end_create(self):
        self.calculate_property()
        self.build_contents()
        self.motions_file.close()
    
        
    def load(self, name):
        self.motions_file = h5py.File(name, 'r')
        for key in self.virtual_attr_list:
            setattr(self, key, self.motions_file['virtual_'+key])
        
    def extract_bvh_with_character(self, name, character, flip = False):
        if flip:
            target = BVHToTargetBase(name, self.fps, character, flip = np.array([1,0,0])).init_target()
        else:
            target = BVHToTargetBase(name, self.fps, character).init_target()
        tarset : SetTargetToCharacter = SetTargetToCharacter(character, target)

        state, ob, done = [],[],[] 
        
        offset = np.zeros(3)
        for i in range(10):
            tarset.set_character_byframe(i)
            aabb = character.get_aabb()
            offset += np.array([0,-aabb[2]+1e-3,0])
        offset /= 10
        
        for i in range(target.num_frames):
            tarset.set_character_byframe(i)
            character.move_character_by_delta(offset)
            state_tmp = character_state(character)
            ob_tmp =  state2ob(torch.from_numpy(state_tmp)).numpy()
            done_tmp = (i == (target.num_frames -1))
            state.append(state_tmp[None,...])
            ob.append(ob_tmp.flatten()[None,...])
            done.append(np.array(done_tmp).reshape(1,1))

        state = np.concatenate(state, axis = 0)
        future = states2future(state)
        ob = np.concatenate(ob, axis = 0)
        
        # if self.clip_feature_set is not None:
        #     new_name = os.path.basename(name)
        #     print(new_name)
        #     # print(self.clip_feature_set.keys())
        #     motion = self.clip_feature_set[new_name]
        #     assert motion.shape[0] == ob.shape[0], (motion.shape[0], ob.shape[0])
        #     clip_feature = np.array(motion)
        
        return {
            'state': state,
            'observation': ob,
            'done': done,
            'future': future,
            # 'clip_feature': clip_feature
        }
    
    def add_bvh_with_character(self, name, character, flip = False):
        motion_name = os.path.basename(name)
        motion_name = motion_name.split('.')[0]
        
        if flip:
            motion_name += '_flip'
        if motion_name in self.motions_file.keys():
            return
        motion = self.extract_bvh_with_character(name, character, flip = flip)
        motion_group = self.motions_file.create_group(motion_name)
        for key in motion.keys():
            motion_group.create_dataset(key, data = motion[key])
        

    def cal_stastics(self, motion):
        
        # print(motion.keys())
        observation = motion['observation']
        future = motion['future']
        
        obs_mean = np.mean(observation, axis=0)
        obs_m2 = (np.std(observation, axis =0)**2) * observation.shape[0]
        
        delta = observation[1:] - observation[:-1]
        _,_,vel, avel,_,_= decompose_obs(delta)
        num = delta.shape[0]
        delta = np.concatenate([vel.reshape(num,-1,3),avel.reshape(num,-1,3)], axis = -1)
        delta = delta.reshape(num,-1)
        delta_mean = np.mean(delta, axis = 0)
        delta_m2 = (np.std(delta, axis = 0)**2) * delta.shape[0]
        future_mean = np.mean(future, axis = 0)
        future_m2 = (np.std(future, axis = 0))**2 * future.shape[0]
        return {
            'num': observation.shape[0],
            'obs_mean': obs_mean,
            'obs_m2': obs_m2,
            'delta_mean': delta_mean,
            'delta_m2': delta_m2,
            'future_mean': future_mean,
            'future_m2': future_m2
        }
    
    def add_folder_bvh(self, name, character, mirror_augment = False, filter = None):
        """Add every bvh in a forlder into motion dataset

        Args:
            name (str): path of bvh folder
            character (ODECharacter): the character of ode
            mirror_augment (bool, optional): whether to use mirror augment. Defaults to True.
        """                
        for file in os.listdir(name):
            if '.bvh' in file:
                if 'normal' in file or 'mill' in file:
                    continue
                print(f'add {file}')
                try:
                    if self.clip_feature_set is not None:
                        new_name = os.path.basename(file)
                        try:
                            motion = self.clip_feature_set[new_name]
                        except:
                            print(new_name, ' not in clip_set, skip')
                            # continue
                    self.add_bvh_with_character(os.path.join(name, file), character)
                except:
                    print('error in ', file)
                    traceback.print_exc()
        
        # if mirror_augment:
        #     for file in os.listdir(name):
        #         if '.bvh' in file:
        #             print(f'add {file} flip')
        #             if self.clip_feature_set is not None:
        #             new_name = os.path.basename(name)
        #             try:
        #                 motion = self.clip_feature_set[new_name]
        #             except:
        #                 print(new_name, '_flip not in clip_set, skip')
        #                 continue
        #             self.add_bvh_with_character(os.path.join(name, file), character, flip = True)
    
    @staticmethod
    def parallel_welford(num_a, avg_a, M2_a, num_b, avg_b, M2_b):
        '''
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        '''
        num = num_a + num_b
        delta = avg_b - avg_a
        M2 = M2_a + M2_b + delta**2 * num_a * num_b / num
        avg = avg_a + delta * num_b / num
        return num, avg, M2


    def calculate_property(self):
        # mean and std
        property = {
            'num': 0,
        }
        cum_num = []
        name_list = ['obs', 'delta', 'future']
        
        motion_names = []
        for motion_name in self.motions_file.keys():
            
            motion = self.motions_file[motion_name]
            
            new_property = self.cal_stastics(motion)
            for name in name_list:
                if name+'_mean' not in property:
                    property[name+'_mean'] = new_property[name+'_mean']
                    property[name+'_m2'] = new_property[name+'_m2']
                else:
                    _, property[name+'_mean'], property[name+'_m2'] = self.parallel_welford(
                        property['num'], property[name+'_mean'], property[name+'_m2'],
                        new_property['num'], new_property[name+'_mean'], new_property[name+'_m2']
                    )
            property['num'] += new_property['num']
            cum_num.append(property['num'])
            motion_names.append(motion_name)
        
        for name in name_list:
            property[name+'_std'] = np.sqrt(property[name+'_m2'] / property['num'])
            property[name+'_std'] = np.clip(property[name+'_std'], 0.1, 1e6)
        
        self.motions_file.create_group('property')
        for key in property.keys():
            if key != 'num':
                self.motions_file['property'].create_dataset(key, data = property[key])
        self.motions_file['property'].attrs['num'] = property['num']
        print('total num: ', property['num'])
        print('total length', property['num'] /20 /60, 'min')
        
        self.motions_file.create_dataset('cum_num', data = np.array([0]+cum_num))
        self.motions_file.create_dataset('motion_name', shape= (len(motion_names), ), dtype = h5py.string_dtype() )
        for i, name in enumerate(motion_names):
            self.motions_file['motion_name'][i] = name
        # self.cum_num = [0]+cum_num
        # self.motion_name = motion_names
        # self.statics = property
    
    @property
    def cum_num(self):
        return self.motions_file['cum_num']
    
    @property
    def motion_name(self):
        return self.motions_file['motion_name']
    
    @property
    def stastics(self):
        return self.motions_file['property']
    
    @property
    def virtual_attr_list(self):
        return ['state', 'observation', 'done', 'future']
    
    def build_contents(self):
        
        attribute_list = self.virtual_attr_list
        for attribute in attribute_list:
            num = self.stastics.attrs['num']
            shape = self.motions_file[self.motion_name[0]][attribute].shape
            layout = h5py.VirtualLayout(shape = (num, *shape[1:]), dtype = self.motions_file[self.motion_name[0]][attribute].dtype)
            for i, motion_name in enumerate(self.motion_name):
                source = h5py.VirtualSource(self.motions_file[motion_name][attribute])
                layout[self.cum_num[i]:self.cum_num[i+1]] = source
            self.motions_file.create_virtual_dataset('virtual_'+attribute, layout, fillvalue = 0)
            setattr(self, attribute, self.motions_file['virtual_'+attribute])  