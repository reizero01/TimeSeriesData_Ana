import pandas as pd
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
from datetime import timedelta, date
from scipy.fftpack import fft, ifft, fftfreq, fftshift
import pickle
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import SGD
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class Fourier:
    def __init__(self):
        # self.data_file = data_file
        self.gauss_sigma = 100
        self.left_steps = []
        self.right_steps = []
        self.data = None
        self.data_set_car = pd.DataFrame()
        self.data_set_metro = pd.DataFrame()
        self.data_set_walking = pd.DataFrame()
        self.data_set_still = pd.DataFrame()
        self.data_set_bus = pd.DataFrame()
        self.car_fft = []
        self.car_fft_real = []
        self.car_fft_img = []
        self.car_fft_radius = []
        self.metro_fft = []
        self.metro_fft_real = []
        self.metro_fft_img = []
        self.bus_fft = []
        self.bus_fft_real = []
        self.bus_fft_img = []
        self.walking_fft = []
        self.walking_fft_real = []
        self.walking_fft_img = []
        self.still_fft = []
        self.still_fft_real = []
        self.still_fft_img = []
        self.KNN_ser = pd.DataFrame()
        self.car_fft_LAcc = []
        self.metro_fft_LAcc = []
        self.bus_fft_LAcc = []
        self.walking_fft_LAcc = []
        self.still_fft_LAcc = []
        self.car_fft_Gyro = []
        self.metro_fft_Gyro = []
        self.bus_fft_Gyro = []
        self.walking_fft_Gyro = []
        self.still_fft_Gyro = []
        self.car_fft_Mag = []
        self.metro_fft_Mag = []
        self.bus_fft_Mag = []
        self.walking_fft_Mag = []
        self.still_fft_Mag = []


    def input_data_set(self):
        dat = ''
        data_set = ['Test_Set/2019_9_24_17_22_35_car_SensorData.csv', 'Test_Set/2019_9_25_23_34_3_metro_SensorData.csv', 'Test_Set/2019_9_26_21_44_59_bus_SensorData.csv', 'Test_Set/2019_9_27_12_0_8_walking_SensorData.csv', 'Test_Set/2019_9_27_14_14_41_still_SensorData.csv']
        
        # data_set = ['Test_Set/2019_9_24_17_22_35_car_SensorData.csv', 'Test_Set/2019_9_24_17_38_36_car_SensorData.csv', 'Test_Set/2019_9_25_23_34_3_metro_SensorData.csv', 'Test_Set/2019_9_26_9_14_42_metro_SensorData.csv',
        # 'Test_Set/2019_9_26_21_44_59_bus_SensorData.csv', 'Test_Set/2019_9_26_22_6_10_bus_SensorData.csv', 'Test_Set/2019_9_27_12_0_8_walking_SensorData.csv', 'Test_Set/2019_9_27_15_57_4_walking_SensorData.csv',
        # 'Test_Set/2019_9_27_14_14_41_still_SensorData.csv', 'Test_Set/2019_9_27_15_28_5_still_SensorData.csv']
        
        while dat != 'q' and dat != 'quit':
            print("Type q or quit to terminate")
            dat = input()
            if(dat != 'q' and dat != 'quit'):
                data_set.append(dat)
        print(data_set)
        
        for data in data_set:
            if(data != 'q' and data != 'quit'):
                features = pd.read_csv(data)
                features = features.assign(Car = [0] * len(features))
                features = features.assign(Metro = [0] * len(features))
                features = features.assign(Bus = [0] * len(features))
                features = features.assign(Walking = [0] * len(features))
                features = features.assign(Still = [0] * len(features))
                Survay1 = features.iloc[0, 24]
                features = features.assign(Survay1 = [Survay1] * len(features))
                
                if("car" in data):
                    features = features.assign(Car = [1] * len(features)) 
                    if(self.data_set_car.empty):
                        self.data_set_car = features    
                    else:
                        frames = [self.data_set_car, features]
                        self.data_set_car = pd.concat(frames, ignore_index=True)
                elif("metro" in data):
                    features = features.assign(Metro = [1] * len(features))
                    if(self.data_set_metro.empty):
                        self.data_set_metro = features    
                    else:
                        frames = [self.data_set_metro, features]
                        self.data_set_metro = pd.concat(frames, ignore_index=True)
                elif("bus" in data):
                    features = features.assign(Bus = [1] * len(features))
                    if(self.data_set_bus.empty):
                        self.data_set_bus = features    
                    else:
                        frames = [self.data_set_bus, features]
                        self.data_set_bus = pd.concat(frames, ignore_index=True)
                elif("walking" in data):
                    features = features.assign(Walking = [1] * len(features))
                    if(self.data_set_walking.empty):
                        self.data_set_walking = features    
                    else:
                        frames = [self.data_set_walking, features]
                        self.data_set_walking = pd.concat(frames, ignore_index=True)
                elif("still" in data):
                    features = features.assign(Still = [1] * len(features))
                    if(self.data_set_still.empty):
                        self.data_set_still = features    
                    else:
                        frames = [self.data_set_still, features]
                        self.data_set_still = pd.concat(frames, ignore_index=True)
        print(self.data_set_car.head(5))

    def comp_conv(self):
        
        self.data_set_car['Acc'] = np.sqrt(self.data_set_car['AccX']**2 + self.data_set_car['AccY']**2 + self.data_set_car['AccZ']**2)
        self.data_set_car['LAcc'] = np.sqrt(self.data_set_car['LAccX']**2 + self.data_set_car['LAccY']**2 + self.data_set_car['LAccZ']**2)
        self.data_set_car['Gyro'] = np.sqrt(self.data_set_car['GyroX']**2 + self.data_set_car['GyroY']**2 + self.data_set_car['GyroZ']**2)
        self.data_set_car['Mag'] = np.sqrt(self.data_set_car['MagX']**2 + self.data_set_car['MagY']**2 + self.data_set_car['MagZ']**2)
        
        self.data_set_car['Acc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_car['Acc'], sigma=self.gauss_sigma)
        self.data_set_car['LAcc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_car['LAcc'], sigma=self.gauss_sigma)
        self.data_set_car['Gyro_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_car['Gyro'], sigma=self.gauss_sigma)
        self.data_set_car['Mag_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_car['Mag'], sigma=self.gauss_sigma)
        print(self.data_set_car.head(5))

        self.data_set_metro['Acc'] = np.sqrt(self.data_set_metro['AccX']**2 + self.data_set_metro['AccY']**2 + self.data_set_metro['AccZ']**2)
        self.data_set_metro['LAcc'] = np.sqrt(self.data_set_metro['LAccX']**2 + self.data_set_metro['LAccY']**2 + self.data_set_metro['LAccZ']**2)
        self.data_set_metro['Gyro'] = np.sqrt(self.data_set_metro['GyroX']**2 + self.data_set_metro['GyroY']**2 + self.data_set_metro['GyroZ']**2)
        self.data_set_metro['Mag'] = np.sqrt(self.data_set_metro['MagX']**2 + self.data_set_metro['MagY']**2 + self.data_set_metro['MagZ']**2)
        
        self.data_set_metro['Acc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_metro['Acc'], sigma=self.gauss_sigma)
        self.data_set_metro['LAcc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_metro['LAcc'], sigma=self.gauss_sigma)
        self.data_set_metro['Gyro_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_metro['Gyro'], sigma=self.gauss_sigma)
        self.data_set_metro['Mag_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_metro['Mag'], sigma=self.gauss_sigma)
        print(self.data_set_metro.head(5))

        self.data_set_bus['Acc'] = np.sqrt(self.data_set_bus['AccX']**2 + self.data_set_bus['AccY']**2 + self.data_set_bus['AccZ']**2)
        self.data_set_bus['LAcc'] = np.sqrt(self.data_set_bus['LAccX']**2 + self.data_set_bus['LAccY']**2 + self.data_set_bus['LAccZ']**2)
        self.data_set_bus['Gyro'] = np.sqrt(self.data_set_bus['GyroX']**2 + self.data_set_bus['GyroY']**2 + self.data_set_bus['GyroZ']**2)
        self.data_set_bus['Mag'] = np.sqrt(self.data_set_bus['MagX']**2 + self.data_set_bus['MagY']**2 + self.data_set_bus['MagZ']**2)
        
        self.data_set_bus['Acc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_bus['Acc'], sigma=self.gauss_sigma)
        self.data_set_bus['LAcc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_bus['LAcc'], sigma=self.gauss_sigma)
        self.data_set_bus['Gyro_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_bus['Gyro'], sigma=self.gauss_sigma)
        self.data_set_bus['Mag_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_bus['Mag'], sigma=self.gauss_sigma)
        print(self.data_set_bus.head(5))

        self.data_set_still['Acc'] = np.sqrt(self.data_set_still['AccX']**2 + self.data_set_still['AccY']**2 + self.data_set_still['AccZ']**2)
        self.data_set_still['LAcc'] = np.sqrt(self.data_set_still['LAccX']**2 + self.data_set_still['LAccY']**2 + self.data_set_still['LAccZ']**2)
        self.data_set_still['Gyro'] = np.sqrt(self.data_set_still['GyroX']**2 + self.data_set_still['GyroY']**2 + self.data_set_still['GyroZ']**2)
        self.data_set_still['Mag'] = np.sqrt(self.data_set_still['MagX']**2 + self.data_set_still['MagY']**2 + self.data_set_still['MagZ']**2)
        
        self.data_set_still['Acc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_still['Acc'], sigma=self.gauss_sigma)
        self.data_set_still['LAcc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_still['LAcc'], sigma=self.gauss_sigma)
        self.data_set_still['Gyro_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_still['Gyro'], sigma=self.gauss_sigma)
        self.data_set_still['Mag_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_still['Mag'], sigma=self.gauss_sigma)
        print(self.data_set_still.head(5))

        self.data_set_walking['Acc'] = np.sqrt(self.data_set_walking['AccX']**2 + self.data_set_walking['AccY']**2 + self.data_set_walking['AccZ']**2)
        self.data_set_walking['LAcc'] = np.sqrt(self.data_set_walking['LAccX']**2 + self.data_set_walking['LAccY']**2 + self.data_set_walking['LAccZ']**2)
        self.data_set_walking['Gyro'] = np.sqrt(self.data_set_walking['GyroX']**2 + self.data_set_walking['GyroY']**2 + self.data_set_walking['GyroZ']**2)
        self.data_set_walking['Mag'] = np.sqrt(self.data_set_walking['MagX']**2 + self.data_set_walking['MagY']**2 + self.data_set_walking['MagZ']**2)
        
        self.data_set_walking['Acc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_walking['Acc'], sigma=self.gauss_sigma)
        self.data_set_walking['LAcc_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_walking['LAcc'], sigma=self.gauss_sigma)
        self.data_set_walking['Gyro_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_walking['Gyro'], sigma=self.gauss_sigma)
        self.data_set_walking['Mag_conv'] = scipy.ndimage.filters.gaussian_filter(self.data_set_walking['Mag'], sigma=self.gauss_sigma)
        print(self.data_set_walking.head(5))

    
    def resampling(self):
        data = [self.data_set_car, self.data_set_metro, self.data_set_bus, self.data_set_walking, self.data_set_still]
        # data = [self.data_set_car]
        counter = 0
        for i in data:
            temp = copy.deepcopy(i)
            temp = temp.rename(columns={"Min":"Minute", "Sec":"Second"})
            temp['Second'] = temp['Second'].iloc[0] 
            temp['Second'] = temp['Second'] + temp['Time']
            temp['Minute'] = temp['Minute'].iloc[0] 
            temp['Hour'] = temp['Hour'].iloc[0]
            temp['Datetime'] = pd.to_datetime(temp[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']] , format="%H:%M:%S:%f")
            start_date = temp['Datetime'].iloc[0] 
            end_date = temp['Datetime'].iloc[-1]

            temp = temp.drop(columns=['Time', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Survay1', 'Survay2', 'Mode'])

        

            temp2 = temp.iloc[0:0]
            temp2['Datetime'] = pd.date_range(start=start_date, end=end_date, freq='20ms')
            print(temp2.head(5))
            temp = temp.append(temp2[~temp2.Datetime.isin(temp['Datetime'])], ignore_index = True)
            temp = temp.sort_values(by=['Datetime'])
            temp = temp.reset_index(drop = True)
            print(temp)
            
            temp = temp.interpolate(method='spline', order = 4)
            print(temp)
            temp3 = temp[temp.Datetime.isin(temp2['Datetime'])]
            temp3 = temp3.append(temp[temp.Datetime == end_date]) 
            temp3 = temp3.reset_index(drop = True)
            print(temp3)
            self.bffft = temp3

            delta = timedelta(seconds=1)
            window_delta = timedelta(seconds=1)
            while start_date <= end_date:
                if (start_date + window_delta) < (end_date + delta):
                    temp4 = temp3[(temp3['Datetime'] >= start_date) & (temp3['Datetime'] <= (start_date + window_delta))]
                    if counter == 0:
                        X = fft(temp4['Acc'])
                        self.car_fft.append(X)
                        X = fft(temp4['LAcc'])
                        self.car_fft_LAcc.append(X)
                        X = fft(temp4['Gyro'])
                        self.car_fft_Gyro.append(X)
                        X = fft(temp4['Mag'])
                        self.car_fft_Mag.append(X)
                        
                    elif counter == 1:
                        X = fft(temp4['Acc'])
                        self.metro_fft.append(X)
                        X = fft(temp4['LAcc'])
                        self.metro_fft_LAcc.append(X)
                        X = fft(temp4['Gyro'])
                        self.metro_fft_Gyro.append(X)
                        X = fft(temp4['Mag'])
                        self.metro_fft_Mag.append(X)
                    elif counter == 2:
                        X = fft(temp4['Acc'])
                        self.bus_fft.append(X)
                        X = fft(temp4['LAcc'])
                        self.bus_fft_LAcc.append(X)
                        X = fft(temp4['Gyro'])
                        self.bus_fft_Gyro.append(X)
                        X = fft(temp4['Mag'])
                        self.bus_fft_Mag.append(X)
                    elif counter == 3:
                        X = fft(temp4['Acc'])
                        self.walking_fft.append(X)
                        X = fft(temp4['LAcc'])
                        self.walking_fft_LAcc.append(X)
                        X = fft(temp4['Gyro'])
                        self.walking_fft_Gyro.append(X)
                        X = fft(temp4['Mag'])
                        self.walking_fft_Mag.append(X)
                    elif counter == 4:
                        X = fft(temp4['Acc'])
                        self.still_fft.append(X)
                        X = fft(temp4['LAcc'])
                        self.still_fft_LAcc.append(X)
                        X = fft(temp4['Gyro'])
                        self.still_fft_Gyro.append(X)
                        X = fft(temp4['Mag'])
                        self.still_fft_Mag.append(X)
                start_date += delta
            counter += 1
        print('car fft: ' + str(len(self.car_fft)))
        print('metro fft: ' + str(len(self.metro_fft)))
        print('bus fft: ' + str(len(self.bus_fft)))
        print('walk fft: ' + str(len(self.walking_fft)))
        print('still fft: ' + str(len(self.still_fft)))
    
    def save_object(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_object(self, filename):
        with open(filename, 'rb') as input:
            self = pickle.load(input)
        print(len(self.car_fft_Mag))
        print(len(self.metro_fft_LAcc))
        print(len(self.bus_fft_Gyro))
        
        print(len(self.walking_fft))
        print(len(self.still_fft))
        # plt.plot(self.data_set_walking['Time'], self.data_set_walking['Acc'])
        # plt.show()
        return self


    def Fourier_Analysis(self, n):

        # test = self.car_fft[1]
        # test1 = self.car_fft[0]
        # test = np.concatenate((test, test1, self.car_fft[2]))
        # print(len(test))
        # N = 1503
        # T = 10.0 / 900.0
        # x = np.linspace(0.0, N*T, N)
        # xf = fftfreq(N, T)
        # xf = fftshift(xf)
        # yplot = fftshift(test)
        # yplot1 = fftshift(test1)
        # plt.plot(xf, 1.0/N * np.abs(yplot))
        # plt.show()
        test = [self.car_fft, self.metro_fft, self.bus_fft, self.walking_fft, self.still_fft]
        # s = test[0]
        df = pd.DataFrame(columns= pd.MultiIndex.from_product([['Acc'], ['f_{}'.format(x) for x in range(1, n+1)]]))
        
        counter = 0
        for fft_set in test:
            
            for s in fft_set:
                avg = []
                for i in range(0, n):
                    sub_set = []
                    if(i != n-1):
                        temp = s[i*int(len(s)/n): int(len(s)/n) + i*int(len(s)/n)]
                        for x in range(len(temp)):
                            real = temp[x].real
                            imag = temp[x].imag
                            sub_set.append(np.sqrt(real ** 2 + imag ** 2))
                        avg.append(sum(sub_set)/len(sub_set))

                    else:
                        temp = s[i*int(len(s)/n) : -1]
                        for x in range(len(temp)):
                            real = temp[x].real
                            imag = temp[x].imag
                            sub_set.append(np.sqrt(real ** 2 + imag ** 2))
                        avg.append(sum(sub_set)/len(sub_set))
                
                df.loc[counter] = avg 
                counter += 1
        
        
        test = [self.car_fft_LAcc, self.metro_fft_LAcc, self.bus_fft_LAcc, self.walking_fft_LAcc, self.still_fft_LAcc]
        df_LAcc = pd.DataFrame(columns= pd.MultiIndex.from_product([['LAcc'], ['f_{}'.format(x) for x in range(1, n+1)]]))
        
        counter = 0
        for fft_set in test:
            
            for s in fft_set:
                avg = []
                for i in range(0, n):
                    sub_set = []
                    if(i != n-1):
                        temp = s[i*int(len(s)/n): int(len(s)/n) + i*int(len(s)/n)]
                        for x in range(len(temp)):
                            real = temp[x].real
                            imag = temp[x].imag
                            sub_set.append(np.sqrt(real ** 2 + imag ** 2))
                        avg.append(sum(sub_set)/len(sub_set))

                    else:
                        temp = s[i*int(len(s)/n) : -1]
                        for x in range(len(temp)):
                            real = temp[x].real
                            imag = temp[x].imag
                            sub_set.append(np.sqrt(real ** 2 + imag ** 2))
                        avg.append(sum(sub_set)/len(sub_set))
                df_LAcc.loc[counter] = avg 
                counter += 1

        test = [self.car_fft_Gyro, self.metro_fft_Gyro, self.bus_fft_Gyro, self.walking_fft_Gyro, self.still_fft_Gyro]
        df_Gyro = pd.DataFrame(columns= pd.MultiIndex.from_product([['Gyro'], ['f_{}'.format(x) for x in range(1, n+1)]]))
        
        counter = 0
        for fft_set in test:
            
            for s in fft_set:
                avg = []
                for i in range(0, n):
                    sub_set = []
                    if(i != n-1):
                        temp = s[i*int(len(s)/n): int(len(s)/n) + i*int(len(s)/n)]
                        for x in range(len(temp)):
                            real = temp[x].real
                            imag = temp[x].imag
                            sub_set.append(np.sqrt(real ** 2 + imag ** 2))
                        avg.append(sum(sub_set)/len(sub_set))

                    else:
                        temp = s[i*int(len(s)/n) : -1]
                        for x in range(len(temp)):
                            real = temp[x].real
                            imag = temp[x].imag
                            sub_set.append(np.sqrt(real ** 2 + imag ** 2))
                        avg.append(sum(sub_set)/len(sub_set))
                df_Gyro.loc[counter] = avg 
                counter += 1
        
        test = [self.car_fft_Mag, self.metro_fft_Mag, self.bus_fft_Mag, self.walking_fft_Mag, self.still_fft_Mag]
        df_Mag = pd.DataFrame(columns= pd.MultiIndex.from_product([['Mag'], ['f_{}'.format(x) for x in range(1, n+1)]]))
        
        counter = 0
        for fft_set in test:
            
            for s in fft_set:
                avg = []
                for i in range(0, n):
                    sub_set = []
                    if(i != n-1):
                        temp = s[i*int(len(s)/n): int(len(s)/n) + i*int(len(s)/n)]
                        for x in range(len(temp)):
                            real = temp[x].real
                            imag = temp[x].imag
                            sub_set.append(np.sqrt(real ** 2 + imag ** 2))
                        avg.append(sum(sub_set)/len(sub_set))

                    else:
                        temp = s[i*int(len(s)/n) : -1]
                        for x in range(len(temp)):
                            real = temp[x].real
                            imag = temp[x].imag
                            sub_set.append(np.sqrt(real ** 2 + imag ** 2))
                        avg.append(sum(sub_set)/len(sub_set))
                df_Mag.loc[counter] = avg 
                counter += 1

        result = pd.concat([df, df_LAcc, df_Gyro, df_Mag], axis=1, sort=False)
        # result = df
        result['Mode'] = ['Car'] * len(self.car_fft) + ['Metro'] * len(self.metro_fft) + ['Bus'] * len(self.bus_fft) + ['Walking'] * len(self.walking_fft) + ['Still'] * len(self.still_fft)
        self.KNN_ser = result
        print(result)

    def K_NN(self):
        
        # test = test + self.metro_fft[0:300] + self.bus_fft[0:300] + self.walking_fft[0:300] + self.still_fft[0:300]
        # test = np.asarray(test)
        # nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(test)
        # distances, indices = nbrs.kneighbors(test)
        # print('Distances')
        # print(distances)
        # print('Indices')
        # print(indices)
        
        # plt.plot(test[0].real, test[0].imag, test[1].real, test[1].imag, test[2].real, test[2].imag, test[3].real, test[3].imag, test[299].real, test[299].imag)
        
        # print(self.KNN_ser)
        
        # X = self.KNN_ser.iloc[:, :-1].values
        X = self.KNN_ser.iloc[:, 0:10].values
        # print(X)
        y = self.KNN_ser.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        # print(self.KNN_ser)

    # baseline model
    def create_baseline(self):
        # create model
        model = Sequential()
        model.add(Dense(10, input_dim=10, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def create_fft(self):
        # print(self.walking_fft)
        
        value_set = []
        
        temp = self.metro_fft
        for y in range(len(temp)):
            sub_set = []
            for x in self.metro_fft[y]:
                real = x.real
                imag = x.imag
                sub_set.append(np.sqrt(real ** 2 + imag ** 2))
            value_set.append(sub_set)
        print(len(value_set))
        test_set = []
        for i in value_set:
            origin_point = fftshift(i)
            point = round(len(origin_point)/2)
            test_set.append(origin_point[point])
        # print(test_set)
        print('len:' + str(len(test_set)))
        # N = 946
        # T = 1.0 / 946.0
        # x = np.linspace(0.0, N*T, N)
        # xf = fftfreq(N, T)
        # xf = fftshift(xf)
        # yplot = fftshift(test_set)
        
        # plt.plot(xf, 1.0/N * np.abs(yplot))
        # plt.show()
        plt.plot(test_set)
        plt.show()

    def C_NN(self):
        
        batch_size = 128
        num_classes = 200
        epochs = 5

        Acc = self.KNN_ser.iloc[:, 0:10].values
        Y = self.KNN_ser.iloc[:, -1].values
        print(len(Acc))
        
        encoder = LabelEncoder()
        encoder.fit(['Car', 'Metro', 'Bus', 'Walking', 'Still'])
        encoded_Y = encoder.transform(Y)
        x_train, x_test, y_train, y_test = train_test_split(Acc, encoded_Y, test_size=0.30)
        print(x_train.shape)
        print(y_train.shape)
        
        x_train = x_train.reshape(x_train.shape[0], 5, 2)
        x_test = x_test.reshape(x_test.shape[0], 5, 2) 
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        img_x, img_y = len(x_train), 10
        input_shape = (5, 2) 


        model = Sequential()
        model.add(Conv1D(32, kernel_size=(2), strides= (1),
                        activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=(1), strides=(1)))
        model.add(Conv1D(64, kernel_size=(2), strides= (1),
                        activation='relu'))
        model.add(MaxPooling1D(pool_size=(1), strides=(1)))
        model.add(Conv1D(64, kernel_size=(2), strides= (1),
                        activation='relu'))
        model.add(MaxPooling1D(pool_size=(1), strides=(1)))
        # model.add(Conv1D(64, (2, 1), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(1, 1)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])

        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        # print(score)
        # plt.plot(range(1, 11), history.acc)
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.show()

        # print(encoded_Y[2000])
        # estimator = KerasClassifier(build_fn=self.create_baseline, epochs=100, batch_size=5, verbose=0)
        # kfold = StratifiedKFold(n_splits=2, shuffle=True)
        # results = cross_val_score(estimator, Acc, encoded_Y, cv=kfold)
        # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    def C_NN2(self):

        num_classes = 1000

        Acc = self.KNN_ser.iloc[:, 0:10].values
        Y = self.KNN_ser.iloc[:, -1].values
        encoder = LabelEncoder()
        encoder.fit(['Car', 'Metro', 'Bus', 'Walking', 'Still'])
        encoded_Y = encoder.transform(Y)
        x_train, x_test, y_train, y_test = train_test_split(Acc, encoded_Y, test_size=0.30)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        model.add(Dense(64, activation='relu', input_dim=10))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

        model.fit(x_train, y_train,
                epochs=10,
                batch_size=128, verbose=0, validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, batch_size=128)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def K_Means_Clustering(self):
        data = self.KNN_ser.iloc[:, 0:10]
        data['Mode'] = self.KNN_ser.iloc[:, -1].values
        print(data)
        car = data['Mode'] == 'Car'
        metro = data['Mode'] == 'Metro'
        bus = data['Mode'] == 'Bus'
        walking = data['Mode'] == 'Walking'
        still = data['Mode'] == 'Still'
        # print(data[still].iloc[0:900, 0:10])
        car = data[bus].iloc[:, 0:10]
        # print(car.iloc[:, 0:1].values)
        print(car[('Acc','f_1')].values)
        df = pd.DataFrame({
            'f1': car[('Acc','f_1')].values,
            'f2': car[('Acc','f_2')].values,
            'f3': car[('Acc','f_3')].values,
            'f4': car[('Acc','f_4')].values,
            'f5': car[('Acc','f_5')].values,
            'f6': car[('Acc','f_6')].values,
            'f7': car[('Acc','f_7')].values,
            'f8': car[('Acc','f_8')].values,
            'f9': car[('Acc','f_9')].values,
            'f10': car[('Acc','f_10')].values,
            
            })
        print(df)
        num_clusters = 1
        kmeans = KMeans(n_clusters=1).fit(df)
        centers = np.array(kmeans.cluster_centers_)
        m_clusters = kmeans.labels_.tolist()
        print(centers)

        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df)
        print(closest)

        # print(car.loc[901].values)

        # print(self.car_fft)

        # N = 501
        # T = 10.0 / 900.0
        # x = np.linspace(0.0, N*T, N)
        # xf = fftfreq(N, T)
        # xf = fftshift(xf)
        # yplot = fftshift(self.bus_fft[closest[0]])
        # yplot1 = fftshift(self.walking_fft[closest[1]])
        # plt.plot(xf, 1.0/N * np.abs(yplot))
        # plt.plot(xf, 1.0/N * np.abs(yplot1))
        # plt.show()

        clostest_data = []
        for i in range(num_clusters):
            center_vec = centers[i]
            data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]

            one_cluster_tf_matrix = np.zeros( (  len(pmids_idx_in_i_cluster) , centers.shape[1] ) )
            for row_num, data_idx in enumerate(data_idx_in_i_cluster):
                one_row = tf_matrix[data_idx]
                one_cluster_tf_matrix[row_num] = one_row

            closest, _ = pairwise_distances_argmin_min(center_vec, one_cluster_tf_matrix)
            closest_idx_in_one_cluster_tf_matrix = closest[0]
            closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
            data_id = all_data[closest_data_row_num]

            closest_data.append(data_id)

        closest_data = list(set(closest_data))

        assert len(closest_data) == num_clusters

        d = kmeans.transform(car['Acc'])[:, 1]
        ind = np.argsort(df)[::-1][:50]
        print(ind)
        plt.scatter(df['car'], df['metro'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=50)
        plt.show()

if __name__ == '__main__':
    driver = Fourier()
    # driver.input_data_set()
    # driver.comp_conv()
    # driver.resampling()
    # driver.save_object('fft_sample_1.pkl')
    # driver = driver.load_object('fft_sample.pkl')
    # driver.create_fft()
    # driver = driver.load_object('fft_sample_1.pkl')
    # driver.create_fft()
    # driver.Fourier_Analysis(10)
    # driver.save_object('KNN_sample_Acc.pkl')
    driver = driver.load_object('KNN_sample.pkl')
    driver.create_baseline()
    # driver.C_NN()
    # driver.C_NN2()
    # driver.K_NN()
    driver.K_Means_Clustering()
    
