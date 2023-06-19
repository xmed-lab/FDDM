import numpy as np
import cv2

class Augmentation(object):
    def resize(self,im,size=(512,512)):
        if im.dtype != 'float32':
            im = im.astype(np.float32)
        im_min, im_max = np.min(im),np.max(im)
        im_std = (im - im_min) / (im_max - im_min)
        resized_std = cv2.resize(im_std, size)
        resized_im = resized_std * (im_max - im_min) + im_min
        return resized_im

    def rotate(self,im,rotation_param):
        h,w,_ = im.shape
        cX,cY = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cX,cY),-rotation_param,1.0)
        cos = np.abs(M[0,0])
        sin = np.abs(M[0,1])

        nW = int(h*sin + w*cos)
        nH = int(h*cos + w*sin)
        M[0,2] += (nW // 2) - cX
        M[1,2] += (nH // 2) - cY

        im_new = cv2.warpAffine(np.array(im,dtype=np.float32),M,(nW,nH))
        
        x0 = int(max(0,(nW-w)/2))
        x1 = int(min((nW+w)/2,nW))
        y0 = int(max(0,(nH-h)/2))
        y1 = int(min((nH+h)/2,nH))
        return im_new[y0:y1, x0:x1]

    def flip(self,im):
        return cv2.flip(im,1)

    def zoom(self,im,zoom_range):
        zoom_min, zoom_max = zoom_range
        h,w,_ = im.shape
        w_dev = int(np.random.uniform(zoom_min, zoom_max) / 2 * w)
        #TODO
        #h_dev = int(..) keep aspect ratio
        return im[w_dev:h-w_dev, w_dev:w-w_dev]

    def crop(self,im,crop_w,crop_h):
        h,w,_ = im.shape
        w_dev = int(crop_w * w)
        h_dev = int(crop_h * h)

        
        w0 = np.random.randint(0, w_dev + 1)
        w1 = np.random.randint(0, w_dev + 1)
        h0 = np.random.randint(0, h_dev + 1)
        h1 = np.random.randint(0, h_dev + 1)

        return im[h0:h-h1, w0:w-w1]

    def black_border_crop(self,im):
        h,w,_ = im.shape

        x_thresh = np.where(np.max(np.max(np.asarray(im),axis=2),axis=0) > 10)[0]
        y_thresh = np.where(np.max(np.max(np.asarray(im),axis=2),axis=1) > 10)[0]
        
        if len(x_thresh) > w // 2:
            min_x, max_x = x_thresh[0],x_thresh[-1]
        else:
            min_x, max_x = 0, w
        if len(y_thresh) > h //2:
            min_y, max_y = y_thresh[0],y_thresh[-1]
        else:
            min_y, max_y = 0, h

        return im[min_y:max_y, min_x:max_x]


    def enhance(self,im): 
        lab= cv2.cvtColor(np.array(im), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20,30))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        im_new = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return im_new

    def clip(self,im):
        im[im>255] = 255
        im[im<0] = 0
        return im

    def brightness(self,im,alpha):
        im *= alpha
        return im

    def contrast(self,im,alpha):
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = im * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        im *= alpha
        im += gray
        return im
   
    def saturation(self,im,alpha):
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = im * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        im *= alpha
        im += gray
        return im

    def multiple_rgb(self,im,alphas):
        for i in xrange(len(alphas)):
            im[:,:,[i]] *= alphas[i]
        return im        

    
class Normlization(object):    
    @staticmethod
    def per_image_norm(img):
        im = np.copy(img)
        if im.dtype != 'float32':
            im = im.astype(np.float32)
        n = im.shape[0] * im.shape[1]*3
        mean = im.sum(axis=(0,1,2)) / n
        s = np.sum(np.square(im),axis=(0,1,2))
        s1 = np.sum(im,axis=(0,1,2))
        std = max(10e-7,np.sqrt((s - s1**2/n) / (n - 1)))
        return np.divide(np.subtract(im,mean),std)

    @staticmethod
    def rgb_norm(im):
        if im.dtype != 'float32':
            im = im.astype(np.float32)
        n = im.shape[0] * im.shape[1]
        mean = im.sum(axis=(0,1)) / n
        s = np.sum(np.square(im),axis=(0,1))
        s1 = np.sum(im,axis=(0,1))
        std = np.sqrt((s - s1**2/n) / (n - 1))
        return np.divide(np.subtract(im,mean),std)

    @staticmethod
    def subtract_mean_norm(im):
        if im.dtype != 'float32':
            im = im.astype(np.float32)
        n = im.shape[0] * im.shape[1]*3
        mean = im.sum(axis=(0,1,2)) / n
        return np.subtract(im, mean)

    @staticmethod
    def divide_255_norm(im):
        if im.dtype != 'float32':
            im = im.astype(np.float32)
        return np.divide(im,255)

class OurAug(Augmentation):
   
    def __init__(self, params, rand=True):
        self.params = params
        self.rand = rand

    def process(self,img):
        im = np.copy(img)
        output_shape = tuple(self.params['output_shape'])
        if  self.params.get('rotation',False):
            rotate_params = np.random.randint(self.params['rotation_range'][0],
                                self.params['rotation_range'][1])
            
            im = self.rotate(im,rotate_params)

        if self.params.get('crop',False):
            do_crop = self.params['crop_prob'] > np.random.rand()

            if do_crop:
                im = self.crop(im,self.params['crop_w'],self.params['crop_h'])

        if self.params.get('flip',False):
            do_flip = self.params['flip_prob'] > np.random.rand()
           
            if do_flip:
                im = self.flip(im)
               
        if self.params.get('zoom',False):
            do_zoom = self.params['zoom_prob'] > np.random.rand()
           
            if do_zoom:
                im = self.zoom(im,self.params['zoom_range'])
        
        # normlize
        #im = Normlization.per_image_norm(im)

        if self.params.get('contrast',False):
            contrast_param = np.random.uniform(self.params['contrast_range'][0],
                                self.params['contrast_range'][1])
            im = self.contrast(im,contrast_param)            


        if self.params.get('brightness',False):
            brightness_param = np.random.uniform(self.params['brightness_range'][0],
                                self.params['brightness_range'][1])
            im = self.brightness(im,brightness_param)            

        if self.params.get('saturation',False):
            color_param = np.random.uniform(self.params['color_range'][0],
                                self.params['color_range'][1])
            im = self.saturation(im,color_param)  

        if self.params.get('multiple_rgb',False):
            multiple_param = tuple(np.random.uniform(self.params['multiple_range'][0],
                                            self.params['multiple_range'][1],3))
            im = self.multiple_rgb(im,multiple_param)

        if tuple(im.shape[:2]) != output_shape:
            im = self.resize(im,output_shape)

        return im
