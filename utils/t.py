import fid_score

if __name__ == '__main__':
    fid_score.get_fid(['./Output/701/',
                    ['../Datasets/CUFS/AR/photos', '../Datasets/CUFS/CUHK/photos', '../Datasets/CUFS/XM2VTS/photos']])
