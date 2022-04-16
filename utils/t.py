import fid_score

if __name__ == '__main__':
    fid = fid_score.get_fid(['./Output/1/50',
                    ['../Datasets/CUFS/AR/photos', '../Datasets/CUFS/CUHK/photos', '../Datasets/CUFS/XM2VTS/photos']])
    print(fid)
