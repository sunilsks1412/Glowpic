# -*- coding: utf-8 -*-


import cv2
import numpy as np
from guidedfilter import guided_filter
from PIL import Image


class illuminate():
        # print("asdf:    ",ig)
        def get_illumination_channel(self, I, w):
            M, N, _ = I.shape
            padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')
            darkch = np.zeros((M, N))
            brightch = np.zeros((M, N))

            for i, j in np.ndindex(darkch.shape):
                darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])
                brightch[i, j] = np.max(padded[i:i + w, j:j + w, :])

            return darkch, brightch

        def get_atmosphere(self, I, brightch, p=0.1):
            M, N = brightch.shape
            flatI = I.reshape(M*N, 3)
            flatbright = brightch.ravel()

            searchidx = (-flatbright).argsort()[:int(M*N*p)]
            A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
            return A

        def get_initial_transmission(self, A, brightch):
            A_c = np.max(A)
            init_t = (brightch-A_c)/(1.-A_c)
            return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t))

        def get_corrected_transmission(self, I, A, darkch, brightch, init_t, alpha, omega, w):
            im3 = np.empty(I.shape, I.dtype);
            for ind in range(0, 3):
                im3[:, :, ind] = I[:, :, ind] / A[ind]
            dark_c, _ = self.get_illumination_channel(im3, w)
            dark_t = 1 - omega*dark_c
            corrected_t = init_t
            diffch = brightch - darkch

            for i in range(diffch.shape[0]):
                for j in range(diffch.shape[1]):
                    if(diffch[i, j] < alpha):
                        corrected_t[i, j] = dark_t[i, j] * init_t[i, j]

            return np.abs(corrected_t)

        def get_final_image(self, I, A, refined_t, tmin):
            refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3))
            J = (I-A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A

            return (J - np.min(J))/(np.max(J) - np.min(J))

        def dehaze(self, I, tmin, w, alpha, omega, p, eps, reduce=False):
            m, n, _ = I.shape
            Idark, Ibright = self.get_illumination_channel(I, w)
            A = self.get_atmosphere(I, Ibright, p)

            init_t = self.get_initial_transmission(A, Ibright) 
            if reduce:
                init_t = self.reduce_init_t(init_t)
            corrected_t = self.get_corrected_transmission(I, A, Idark, Ibright, init_t, alpha, omega, w)

            normI = (I - I.min()) / (I.max() - I.min())
            refined_t = guided_filter(normI, corrected_t, w, eps)
            J_refined = self.get_final_image(I, A, refined_t, tmin)
            
            enhanced = (J_refined*255).astype(np.uint8)
            f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
            f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
            return f_enhanced

        def reduce_init_t(self, init_t):
            init_t = (init_t*255).astype(np.uint8)
            xp = [0, 32, 255]
            fp = [0, 32, 48]
            x = np.arange(256)
            table = np.interp(x, xp, fp).astype('uint8')
            init_t = cv2.LUT(init_t, table)
            init_t = init_t.astype(np.float64)/255
            return init_t

        # cv2.imshow("sapm", ig)
        def starter(self, ig):
            print("sample:    ", type(ig))
            # im = cv2.imread(ig)
            im = Image.fromarray(ig)
            orig = im.copy()

            tmin = 0.1   # minimum value for t to make J image
            w = 15       # window size, which determine the corseness of prior images
            alpha = 0.4  # threshold for transmission correction
            omega = 0.75 # this is for dark channel prior
            p = 0.1      # percentage to consider for atmosphere
            eps = 1e-3   # for J image

            I = np.asarray(im, dtype=np.float64) # Convert the input to an array.
            I = I[:, :, :3] / 255

            f_enhanced = self.dehaze(I, tmin, w, alpha, omega, p, eps)
            f_enhanced2 = self.dehaze(I, tmin, w, alpha, omega, p, eps, True)
            
            cv2.imwrite(f"f2{I[3,4,2]}.jpg", f_enhanced2)
            
            return f_enhanced2
            
        # cv2.imshow('original', orig)
        # cv2.imshow('F_enhanced', f_enhanced)
        # cv2.imshow('F_enhanced2', f_enhanced2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
