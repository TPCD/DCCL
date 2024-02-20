import copy
import os
import os.path as osp
import re

import matplotlib.pyplot as plt
import numpy as np
# from MulticoreTSNE import MulticoreTSNE as TSNE

# from tsnecuda import TSNE
# string = 'abe(ac)ad)'
p1 = re.compile(r'[(](.*?)[)]', re.S)  # 最小匹配
import seaborn as sns
from PIL import Image


import time
def time_now():
    '''return current time in format of 2000-01-01 12:01:01'''
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plot_with_labels(save_path, lowDWeights, labels, camid):
    import matplotlib.pyplot as plt
    padding_rate = 0.1
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    unique = np.unique(labels)
    num_type_color = len(unique)
    cmap = get_cmap(num_type_color)
    label2color_dict = {l: cmap(i) for i, l in enumerate(unique)}

    for x, y, s, cam in zip(X, Y, labels, camid):
        if cam == 9:
            plt.text(x, y, s, fontsize=3,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        elif cam == 8:
            plt.text(x, y, s, fontsize=3,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='square',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        elif cam == 7:
            plt.text(x, y, s, fontsize=5,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        elif cam == 3 or cam == 6:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        elif cam == 0:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        else:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='square',  #
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
    # plt.xlim(X.min()-padding_rate*X.min(), X.max()+padding_rate*X.max())
    # plt.ylim(Y.min()-padding_rate*Y.min(), Y.max()+padding_rate*Y.max())
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.axis('off')

    plt.title('Visualize Embedding')
    save_path = save_path + '.png'
    plt.savefig(save_path, dpi=800)
    plt.show()

    plt.pause(0.01)


def plot_with_domains(save_path, lowDWeights, labels):
    import matplotlib.pyplot as plt
    padding_rate = 0.1
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    unique = np.unique(labels)

    num_type_color = 7
    colors = sns.color_palette("viridis", n_colors=num_type_color)
    # label2color_dict = {l: cmap(i) for i, l in enumerate(unique)}
    box_alpha = 0.3
    for x, y, s in zip(X, Y, labels):
        if s >= 0 and s < 100:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[0],  # 边框颜色
                               fc=colors[0]  # 填充颜色
                               ))
        elif s >= 100 and s < 200:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[1],  # 边框颜色
                               fc=colors[1]  # 填充颜色
                               ))
        elif s >= 200 and s < 300:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[2],  # 边框颜色
                               fc=colors[2]  # 填充颜色
                               ))
        elif s >= 300 and s < 400:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[3],  # 边框颜色
                               fc=colors[3]  # 填充颜色
                               ))
        elif s >= 400 and s < 500:
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[4],  # 边框颜色
                               fc=colors[4]  # 填充颜色
                               ))
        else:
            plt.text(x, y, s, fontsize=5,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[5],  # 边框颜色
                               fc=colors[5]  # 填充颜色
                               ))
    # plt.xlim(X.min()-padding_rate*X.min(), X.max()+padding_rate*X.max())
    # plt.ylim(Y.min()-padding_rate*Y.min(), Y.max()+padding_rate*Y.max())
    # plt.legend()
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.axis('off')

    # plt.title('Visualize Embedding')
    save_path = save_path + '.png'
    plt.savefig(save_path, dpi=800)
    plt.show()

    plt.pause(0.01)


def plot_with_domains_and_pic(save_path, lowDWeights, labels, pic_path):
    import matplotlib.pyplot as plt
    padding_rate = 0.1
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    unique = np.unique(labels)

    num_type_color = 7
    colors = sns.color_palette("viridis", n_colors=num_type_color)
    # label2color_dict = {l: cmap(i) for i, l in enumerate(unique)}
    box_alpha = 0.3
    for x, y, s, path in zip(X, Y, labels, pic_path):
        # path = f'{path.split("/")}'
        path = path.replace('/data/ReIDDatasets/', '/home/r2d2/r2d2/Datasets/')
        if s >= 0 and s < 100:
            im = np.array(Image.open(path))
            plt.imshow(im, extent=[x, x + 0.2, y, y + 0.4], aspect='auto', cmap='gray')
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[0],  # 边框颜色
                               fc=colors[0]  # 填充颜色
                               ))
        elif s >= 100 and s < 200:
            im = np.array(Image.open(path))
            plt.imshow(im, extent=[x, x + 0.2, y, y + 0.4], aspect='auto', cmap='gray')
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[1],  # 边框颜色
                               fc=colors[1]  # 填充颜色
                               ))
        elif s >= 200 and s < 300:
            im = np.array(Image.open(path))
            plt.imshow(im, extent=[x, x + 0.2, y, y + 0.4], aspect='auto', cmap='gray')
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[2],  # 边框颜色
                               fc=colors[2]  # 填充颜色
                               ))
        elif s >= 300 and s < 400:
            im = np.array(Image.open(path))
            plt.imshow(im, extent=[x, x + 0.2, y, y + 0.4], aspect='auto', cmap='gray')
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[3],  # 边框颜色
                               fc=colors[3]  # 填充颜色
                               ))
        elif s >= 400 and s < 500:
            im = np.array(Image.open(path))
            plt.imshow(im, extent=[x, x + 0.2, y, y + 0.4], aspect='auto', cmap='gray')
            plt.text(x, y, s, fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[4],  # 边框颜色
                               fc=colors[4]  # 填充颜色
                               ))
        else:
            plt.text(x, y, s, fontsize=2,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle', alpha=box_alpha,  # 圆圈
                               ec=colors[5],  # 边框颜色
                               fc=colors[5]  # 填充颜色
                               ))
    # plt.xlim(X.min()-padding_rate*X.min(), X.max()+padding_rate*X.max())
    # plt.ylim(Y.min()-padding_rate*Y.min(), Y.max()+padding_rate*Y.max())
    # plt.legend()
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.axis('off')

    # plt.title('Visualize Embedding')
    save_path = save_path + '.png'
    plt.savefig(save_path, dpi=800)
    plt.show()

    plt.pause(0.01)


def plot_2D_embedding(save_path, lowDWeights, labels):
    import matplotlib.pyplot as plt
    padding_rate = 0.1
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    unique = np.unique(labels)
    num_type_color = len(unique)
    cmap = get_cmap(num_type_color)
    label2color_dict = {l: cmap(i) for i, l in enumerate(unique)}
    colors = sns.color_palette("viridis", n_colors=num_type_color)

    for x, y, s in zip(X, Y, labels):
        plt.text(x, y, s, fontsize=1,
                 ha='center', va='center',  # 水平居中,垂直居中
                 bbox=dict(boxstyle='square',  #
                           ec=colors[s],  # 边框颜色
                           fc=colors[s]  # 填充颜色
                           ))
    # plt.xlim(X.min()-padding_rate*X.min(), X.max()+padding_rate*X.max())
    # plt.ylim(Y.min()-padding_rate*Y.min(), Y.max()+padding_rate*Y.max())
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.axis('off')

    # plt.title('Visualize Embedding')
    save_path = save_path + '.png'
    plt.savefig(save_path, dpi=800)
    plt.show()

    plt.pause(0.01)



def plot_2D_embedding_mask(save_path, lowDWeights, labels, mask):
    import matplotlib.pyplot as plt
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    unique = np.unique(labels)
    num_type_color = len(unique)
    cmap = get_cmap(num_type_color)
    label2color_dict = {l: i for i, l in enumerate(unique)}
    c_labels = [label2color_dict[l] for l in labels]
    # colors = sns.color_palette("viridis", n_colors=num_type_color)
    colors = sns.color_palette("husl", n_colors=num_type_color)


    for x, y, s, m in zip(X, Y, c_labels, mask):
        if m:
            plt.text(x, y, ' ', fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(alpha=0.5,boxstyle='circle',  #
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        else:

            plt.text(x, y, ' ',  fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(alpha=0.5,boxstyle='circle',  #
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
    # plt.xlim(X.min()-padding_rate*X.min(), X.max()+padding_rate*X.max())
    # plt.ylim(Y.min()-padding_rate*Y.min(), Y.max()+padding_rate*Y.max())
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.axis('off')

    # plt.title('Visualize Embedding')
    save_path = save_path + '.png'
    plt.savefig(save_path, dpi=800)




def plot_without_labels(save_path, lowDWeights, labels, camid):
    import matplotlib.pyplot as plt
    padding_rate = 0.1
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    unique = np.unique(labels)
    num_type_color = len(unique)
    cmap = get_cmap(num_type_color)
    colors = {l: cmap(i) for i, l in enumerate(unique)}
    colors = sns.color_palette("viridis", n_colors=num_type_color)

    for x, y, s, cam in zip(X, Y, labels, camid):
        if cam == 9:
            plt.text(x, y, ' ', fontsize=3,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        elif cam == 8:
            plt.text(x, y, ' ', fontsize=3,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='square',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        elif cam == 7:
            plt.text(x, y, ' ', fontsize=5,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        elif cam == 3 or cam == 6:
            plt.text(x, y, ' ', fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        elif cam == 0:
            plt.text(x, y, ' ', fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='circle',  # 圆圈
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
        else:
            plt.text(x, y, ' ', fontsize=1,
                     ha='center', va='center',  # 水平居中,垂直居中
                     bbox=dict(boxstyle='square',  #
                               ec=colors[s],  # 边框颜色
                               fc=colors[s]  # 填充颜色
                               ))
    plt.xlim(X.min() - padding_rate * X.min(), X.max() + padding_rate * X.max())
    plt.ylim(Y.min() - padding_rate * Y.min(), Y.max() + padding_rate * Y.max())
    # plt.xlim(X.min(), X.max())
    # plt.ylim(Y.min(), Y.max())
    plt.axis('off')

    plt.title('Visualize Embedding')
    save_path = save_path + '.png'
    plt.savefig(save_path, dpi=800)
    plt.show()

    plt.pause(0.01)


def visualize(features_path, output_path, show_num_identities_for_each_domain, plot_by_domain=False, if_from_numpy='',
              if_generalization=False):
    from collections import defaultdict
    datasets = defaultdict(dict)
    if not osp.exists(features_path):
        assert False, f'features_path ({features_path}) is not existing!'
    query_dict_list, gallery_dict_list = [], []
    for _file in sorted(os.listdir(features_path)):
        print(_file)
        if 'cids' in _file or 'fuse' in _file:
            continue
        features = np.load(osp.join(features_path, _file))
        dataset_name = re.findall(p1, _file)[0]
        if 'pids' in _file:
            if 'pids' in datasets[dataset_name].keys():
                datasets[dataset_name]['pids'] = np.concatenate(
                    (datasets[dataset_name]['pids'], features), axis=0)
                # print(f'{dataset_name} ==> add pids: {features.shape}')
            else:
                datasets[dataset_name].update({'pids': features})
                # print(f'{dataset_name} ==> initial pids: {features.shape}')

        elif 'features' in _file:
            if 'features' in datasets[dataset_name].keys():
                datasets[dataset_name]['features'] = np.concatenate(
                    (datasets[dataset_name]['features'], features), axis=0)
                # print(f'{dataset_name} ==> add features: {features.shape}')
            else:
                datasets[dataset_name].update({'features': features})
                # print(f'{dataset_name} ==> initial features: {features.shape}')
        elif 'paths' in _file:
            if 'paths' in datasets[dataset_name].keys():
                datasets[dataset_name]['paths'] = np.concatenate(
                    (datasets[dataset_name]['paths'], features), axis=0)
                # print(f'{dataset_name} ==> add features: {features.shape}')
            else:
                datasets[dataset_name].update({'paths': features})
                # print(f'{dataset_name} ==> initial features: {features.shape}')
        elif 'meta_graph_vertex' in _file:
            meta_graph_vertex = features
        else:
            assert False, f'the name of _file is {_file}, which is not compatible'
    assert meta_graph_vertex is not None
    print(f'meta_graph_vertex is <{meta_graph_vertex.shape}>!')
    print(list(datasets.keys()))
    for k, v in datasets.items():
        # print(k)
        # print(np.max(v['pids']))
        # print(np.min(v['pids']))
        v['pids'] = relabel_numpy(v['pids'])
        # print(np.max(v['pids']))
        # print(np.min(v['pids']))

    for k, v in datasets.items():
        selected_ids = np.random.permutation(np.max(v['pids']))[
                       :show_num_identities_for_each_domain]
        print(f'selected id for {k} is : {selected_ids}')

        # index = np.arange(v['pids'].shape[0])
        # for ii, pid in enumerate(v['pids']):
        #     if pid in selected_ids:
        #         if 'selected_features' and 'selected_pids' in v.keys():
        #             v['selected_features'] = np.concatenate((v['selected_features'], v['features'][ii]), axis=0)
        #             assert v['pids'][ii] == pid, 'conflict'
        #             v['selected_pids'] = np.concatenate((v['selected_pids'], pid), axis=0)
        #         else:
        #             v.update({'selected_features': v['features'][ii]})
        #             assert v['pids'][ii] == pid, 'conflict'
        #             v.update({'selected_pids': pid})
        delete_index = []
        list_selected_ids = selected_ids.tolist()
        for ii, pid in enumerate(v['pids']):
            if pid not in list_selected_ids:
                delete_index.append(ii)
        v['pids'] = np.delete(v['pids'], delete_index, axis=0)
        v['features'] = np.delete(v['features'], delete_index, axis=0)
        v['paths'] = np.delete(v['paths'], delete_index, axis=0)
        v['pids'] = relabel_numpy(v['pids'])
        print(
            f"Domain: ({k}) \t the number of visulised features is ({v['features'].shape})")

    if if_generalization:
        concated_features = datasets['allgeneralizable']['features']
        concated_pids = datasets['allgeneralizable']['pids']
        concated_paths = datasets['allgeneralizable']['paths']
    else:
        list_features, list_pids, list_paths = [], [], []
        # datasets['market']['pids']
        list_features.append(datasets['market']['features'])
        list_pids.append(datasets['market']['pids'])
        list_paths.append(datasets['market']['paths'])
        # datasets['cuhksysu']['pids']
        datasets['cuhksysu']['pids'] += 100
        list_features.append(datasets['cuhksysu']['features'])
        list_pids.append(datasets['cuhksysu']['pids'])
        list_paths.append(datasets['cuhksysu']['paths'])
        # datasets['duke']
        datasets['duke']['pids'] += 200
        list_features.append(datasets['duke']['features'])
        list_pids.append(datasets['duke']['pids'])
        list_paths.append(datasets['duke']['paths'])
        # datasets['msmt17']['pids']
        datasets['msmt17']['pids'] += 300
        list_features.append(datasets['msmt17']['features'])
        list_pids.append(datasets['msmt17']['pids'])
        list_paths.append(datasets['msmt17']['paths'])
        # datasets['cuhk03']['pids']
        datasets['cuhk03']['pids'] += 400
        list_features.append(datasets['cuhk03']['features'])
        list_pids.append(datasets['cuhk03']['pids'])
        list_paths.append(datasets['cuhk03']['paths'])
        concated_features = np.concatenate(list_features, axis=0)
        concated_pids = np.concatenate(list_pids, axis=0)
        concated_paths = np.concatenate(list_paths, axis=0)

    number_vertex = meta_graph_vertex.shape[0]
    vertex_label = np.arange(number_vertex, dtype=np.int64)
    vertex_path_list = []
    for v_l in vertex_label:
        vertex_path_list.append('vertex')
    vertex_label = vertex_label + 10000
    # meta_graph_vertex l2 normal
    # meta_graph_vertex = meta_graph_vertex/np.expand_dims(np.linalg.norm(meta_graph_vertex, axis=1), axis=1)
    # meta_graph_vertex = meta_graph_vertex/np.expand_dims(np.linalg.norm(meta_graph_vertex, axis=1), axis=1)
    # concated_paths = np.concatenate((concated_paths, datasets['market']['paths'][-number_vertex:]), axis=0)
    concated_paths = np.concatenate((concated_paths, np.array(vertex_path_list)), axis=0)
    concated_features = np.concatenate((concated_features, meta_graph_vertex), axis=0)
    concated_pids = np.concatenate((concated_pids, vertex_label), axis=0)
    # concated_features l2 norm
    # concated_features = concated_features/np.expand_dims(np.linalg.norm(concated_features, axis=1), axis=1)
    if if_from_numpy:
        print(f'Time: {time_now} \n Load {if_from_numpy} begin ...')
        path = osp.join(if_from_numpy, 'embedding_2D.npy')
        X_embedded = np.load(path)
    else:
        print(f'Time: {time_now} \n Dimentional Reduction begin ...')
        tsne = TSNE(n_jobs=32)
        X_embedded = tsne.fit_transform(concated_features)
        # X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(concated_features)
        print(f'Time: {time_now} \n Dimentional Reduction end ...')
        output_embedding_path = osp.join(output_path, 'embedding_2D.npy')
        output_paths_path = osp.join(output_path, 'concated_paths.npy')
        output_pids_path = osp.join(output_path, 'concated_pids.npy')
        print(f'Time: {time_now} \n Saving 2D embedding in {output_embedding_path}!')
        np.save(output_embedding_path, X_embedded)
        print(f'Time: {time_now} \n Saving concated_paths in {output_paths_path}!')
        np.save(output_paths_path, concated_paths)
        print(f'Time: {time_now} \n Saving concated_pids in {output_pids_path}!')
        np.save(output_pids_path, concated_pids)
        print(f'Time: {time_now} \n Embedding saved!')
    if plot_by_domain:
        plot_with_domains_and_pic(output_path, X_embedded, concated_pids, concated_paths)
    else:
        plot_2D_embedding(output_path, X_embedded, concated_pids)


def visualize_gcd(features_path, output_path, show_num_identities_for_each_domain, if_reduce_dim_before_select=False, if_from_numpy=''):
    from collections import defaultdict
    datasets = defaultdict(dict)
    if not osp.exists(features_path[0]):
        assert False, f'features_path ({features_path}) is not existing!'
    query_dict_list, gallery_dict_list = [], []
    masks = None
    if if_from_numpy:
        print(f'Time: {time_now} \n Load {if_from_numpy} begin ...')
        path = osp.join(if_from_numpy, 'embedding_2D.npy')
        # saved_target_path = osp.join(if_from_numpy, 'target.npy')
        # saved_mark_path = osp.join(if_from_numpy, 'mark.npy')
        X_embedded = np.load(path)
        targets = np.load(features_path[1])
        masks = np.load(features_path[2])
    else:
        print(f'Time: {time_now} \n Dimentional Reduction begin ...')
        tsne = TSNE(n_jobs=32)
        l2_features = np.load(features_path[0])
        targets = np.load(features_path[1])
        masks = np.load(features_path[2])
        if if_reduce_dim_before_select is False:
            targets = targets.astype(np.int64)
            masks = masks.astype(np.bool)
            old_classes = np.unique(targets[masks])
            new_classes = np.unique(targets[~masks])
            if show_num_identities_for_each_domain == 0:
                selected_old_classes = old_classes
                selected_new_classes = new_classes
            else:
                selected_old_classes = np.random.choice(old_classes, show_num_identities_for_each_domain)
                selected_new_classes = np.random.choice(new_classes, show_num_identities_for_each_domain)
            targets_, masks_ = np.array([]), np.array([])
            selected_feature = None
            for _f, _l, _t in zip(l2_features, targets, masks):
                if _l in selected_old_classes:
                    if selected_feature is None:
                        selected_feature = np.expand_dims(_f, 0)
                    else:
                        selected_feature = np.concatenate((selected_feature, np.expand_dims(_f, 0)), axis=0)
                    targets_ = np.append(targets_, _l)
                    masks_ = np.append(masks_, _t)
                elif _l in selected_new_classes:
                    if selected_feature is None:
                        selected_feature = np.expand_dims(_f, 0)
                    else:
                        selected_feature = np.concatenate((selected_feature, np.expand_dims(_f, 0)), axis=0)
                    targets_ = np.append(targets_, _l)
                    masks_ = np.append(masks_, _t)
        else:
            selected_feature = l2_features
        targets = targets_
        masks = masks_
        X_embedded = tsne.fit_transform(selected_feature)
        # X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(concated_features)
        print(f'Time: {time_now} \n Dimentional Reduction end ...')
        output_embedding_path = osp.join(output_path, 'embedding_2D.npy')
        # output_paths_path = osp.join(output_path, 'concated_paths.npy')
        # output_pids_path = osp.join(output_path, 'concated_pids.npy')
        print(f'Time: {time_now} \n Saving 2D embedding in {output_embedding_path}!')
        np.save(output_embedding_path, X_embedded)
    if if_reduce_dim_before_select is True:
        targets = targets.astype(np.int64)
        masks = masks.astype(np.bool)
        old_classes = np.unique(targets[masks])
        new_classes = np.unique(targets[~masks])
        selected_old_classes = np.random.choice(old_classes, show_num_identities_for_each_domain)
        selected_new_classes = np.random.choice(new_classes, show_num_identities_for_each_domain)
        targets_, masks_ = np.array([]), np.array([])
        X_embedded_ = None
        for _f, _l, _t in zip(X_embedded, targets, masks):
            if _l in selected_old_classes:
                if X_embedded_ is None:
                    X_embedded_ = np.expand_dims(_f, 0)
                else:
                    X_embedded_ = np.concatenate((X_embedded_,np.expand_dims(_f, 0)), axis=0)
                targets_ = np.append(targets_, _l)
                masks_ = np.append(masks_, _t)
            elif _l in selected_new_classes:
                if X_embedded_ is None:
                    X_embedded_ = np.expand_dims(_f, 0)
                else:
                    X_embedded_ = np.concatenate((X_embedded_, np.expand_dims(_f, 0)), axis=0)
                targets_ = np.append(targets_, _l)
                masks_ = np.append(masks_, _t)
        X_embedded = X_embedded_
        targets = targets_
        masks = masks_

    targets = targets.astype(np.int64)
    masks = masks.astype(np.bool)
    output_path = output_path + '/'
    if masks is None:
        plot_2D_embedding(output_path, X_embedded, targets)
    else:
        plot_2D_embedding_mask(output_path, X_embedded, targets, masks)



def visualize2(features_path, output_path, show_num_identities_for_each_domain=100):
    from collections import defaultdict
    datasets = defaultdict(dict)
    if not osp.exists(features_path):
        assert False, f'features_path ({features_path}) is not existing!'
    query_dict_list, gallery_dict_list = [], []
    for _file in sorted(os.listdir(features_path)):
        print(_file)
        if 'cids' in _file:
            continue
        features = np.load(osp.join(features_path, _file))
        dataset_name = re.findall(p1, _file)[0]
        if 'pids' in _file:
            if 'pids' in datasets[dataset_name].keys():
                datasets[dataset_name]['pids'] = np.concatenate(
                    (datasets[dataset_name]['pids'], features), axis=0)
                # print(f'{dataset_name} ==> add pids: {features.shape}')
            else:
                datasets[dataset_name].update({'pids': features})
                # print(f'{dataset_name} ==> initial pids: {features.shape}')

        elif 'features' in _file:
            if 'features' in datasets[dataset_name].keys():
                datasets[dataset_name]['features'] = np.concatenate(
                    (datasets[dataset_name]['features'], features), axis=0)
                # print(f'{dataset_name} ==> add features: {features.shape}')
            else:
                datasets[dataset_name].update({'features': features})
                # print(f'{dataset_name} ==> initial features: {features.shape}')
        else:
            assert False, f'the name of _file is {_file}, which is not compatible'

    print(list(datasets.keys()))
    for k, v in datasets.items():
        # print(k)
        # print(np.max(v['pids']))
        # print(np.min(v['pids']))
        v['pids'] = relabel_numpy(v['pids'])
        # print(np.max(v['pids']))
        # print(np.min(v['pids']))

    for k, v in datasets.items():
        selected_ids = np.random.permutation(np.max(v['pids']))[
                       :show_num_identities_for_each_domain]

        index = np.arange(v['pids'].shape[0])
        # for ii, pid in enumerate(v['pids']):
        #     if pid in selected_ids:
        #         if 'selected_features' and 'selected_pids' in v.keys():
        #             v['selected_features'] = np.concatenate((v['selected_features'], v['features'][ii]), axis=0)
        #             assert v['pids'][ii] == pid, 'conflict'
        #             v['selected_pids'] = np.concatenate((v['selected_pids'], pid), axis=0)
        #         else:
        #             v.update({'selected_features': v['features'][ii]})
        #             assert v['pids'][ii] == pid, 'conflict'
        #             v.update({'selected_pids': pid})
        delete_index = []
        for ii, pid in enumerate(v['pids']):
            if pid not in selected_ids:
                delete_index.append(ii)
        np.delete(v['pids'], delete_index, axis=0)
        np.delete(v['features'], delete_index, axis=0)
        v['pids'] = relabel_numpy(v['pids'])
        print(
            f"Domain: ({k}) \t the number of visulised features is ({v['features'].shape})")

    concated_features = datasets['allgeneralizable']['features']
    concated_pids = datasets['allgeneralizable']['pids']
    print(f'Time: {time_now} \n Dimentional Reduction begin ...')
    tsne = TSNE(n_jobs=32)
    X_embedded = tsne.fit_transform(concated_features)
    # X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(concated_features)
    print(f'Time: {time_now} \n Dimentional Reduction end ...')
    output_file_path = osp.join(output_path, 'embedding_2D.npy')
    print(f'Time: {time_now} \n Saving 2D embedding in {output_file_path}!')
    np.save(output_file_path, X_embedded)
    print(f'Time: {time_now} \n Embedding saved!')
    plot_2D_embedding(output_path, X_embedded, concated_pids)


def visualize3(features_path, output_path, show_num_identities_for_each_domain=100, plot_by_domain=False,
               if_from_numpy='', if_generalization=True):
    from collections import defaultdict
    datasets = defaultdict(dict)
    if not osp.exists(features_path):
        assert False, f'features_path ({features_path}) is not existing!'
    query_dict_list, gallery_dict_list = [], []
    for _file in sorted(os.listdir(features_path)):
        print(_file)
        if 'cids' in _file or 'fuse' in _file:
            continue
        features = np.load(osp.join(features_path, _file))
        dataset_name = re.findall(p1, _file)[0]
        if 'pids' in _file:
            if 'pids' in datasets[dataset_name].keys():
                datasets[dataset_name]['pids'] = np.concatenate(
                    (datasets[dataset_name]['pids'], features), axis=0)
                # print(f'{dataset_name} ==> add pids: {features.shape}')
            else:
                datasets[dataset_name].update({'pids': features})
                # print(f'{dataset_name} ==> initial pids: {features.shape}')

        elif 'features' in _file:
            if 'features' in datasets[dataset_name].keys():
                datasets[dataset_name]['features'] = np.concatenate(
                    (datasets[dataset_name]['features'], features), axis=0)
                # print(f'{dataset_name} ==> add features: {features.shape}')
            else:
                datasets[dataset_name].update({'features': features})
                # print(f'{dataset_name} ==> initial features: {features.shape}')
        else:
            assert False, f'the name of _file is {_file}, which is not compatible'

    print(list(datasets.keys()))
    for k, v in datasets.items():
        # print(k)
        # print(np.max(v['pids']))
        # print(np.min(v['pids']))
        v['pids'] = relabel_numpy(v['pids'])
        # print(np.max(v['pids']))
        # print(np.min(v['pids']))

    for k, v in datasets.items():
        # selected_ids = np.random.permutation(np.max(v['pids']))[
        #     :show_num_identities_for_each_domain]
        selected_ids = np.random.permutation(np.max(v['pids']))

        index = np.arange(v['pids'].shape[0])
        # for ii, pid in enumerate(v['pids']):
        #     if pid in selected_ids:
        #         if 'selected_features' and 'selected_pids' in v.keys():
        #             v['selected_features'] = np.concatenate((v['selected_features'], v['features'][ii]), axis=0)
        #             assert v['pids'][ii] == pid, 'conflict'
        #             v['selected_pids'] = np.concatenate((v['selected_pids'], pid), axis=0)
        #         else:
        #             v.update({'selected_features': v['features'][ii]})
        #             assert v['pids'][ii] == pid, 'conflict'
        #             v.update({'selected_pids': pid})
        delete_index = []
        for ii, pid in enumerate(v['pids']):
            if pid not in selected_ids:
                delete_index.append(ii)
        v['pids'] = np.delete(v['pids'], delete_index, axis=0)
        v['features'] = np.delete(v['features'], delete_index, axis=0)
        v['pids'] = relabel_numpy(v['pids'])
        print(
            f"Domain: ({k}) \t the number of visulised features is ({v['features'].shape})")

    concated_features = datasets['allgeneralizable']['features']
    concated_pids = datasets['allgeneralizable']['pids']
    print(f'Time: {time_now()} \n loading Dimentional Reduction ...')
    tsne = TSNE(n_jobs=32)
    X_embedded = tsne.fit_transform(concated_features)
    X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(concated_features)
    print(f'Time: {time_now()} \n Dimentional Reduction end ...')
    output_file_path = osp.join(
        '/data/pun/results/embedding', 'embedding_2D.npy')
    print(f'Time: {time_now()} \n Saving 2D embedding in {output_file_path}!')
    np.save(output_file_path, X_embedded)
    # X_embedded = np.load(output_file_path)
    plot_2D_embedding(output_path, X_embedded, concated_pids)
    print(f'Time: {time_now()} \n Finished visulization ...')


def relabel_numpy(pids_array, if_reture_dict=False):
    assert len(
        pids_array.shape) == 1, f'The shape of np_array is {pids_array.shape}!'
    pids_np_array = copy.deepcopy(pids_array)
    label_dict = {}
    for i, original_label in enumerate(np.unique(pids_np_array)):
        label_dict[original_label] = i

    new_pids = np.array([label_dict[original_label]
                         for original_label in pids_np_array])
    if if_reture_dict:
        return new_pids, label_dict
    else:
        return new_pids



""" Logging to Visdom server """
import numpy as np
import visdom
import torchvision as tv

class Logger:

    def __init__(self, log_file):
        '''/path/to/log_file.txt'''
        self.log_file = log_file

    def __call__(self, input):
        input = str(input)
        with open(self.log_file, 'a') as f:
            f.writelines(input+'\n')
        print(input)



class BaseVisdomLogger(Logger):
    '''
        The base class for logging output to Visdom.

        ***THIS CLASS IS ABSTRACT AND MUST BE SUBCLASSED***

        Note that the Visdom server is designed to also handle a server architecture,
        and therefore the Visdom server must be running at all times. The server can
        be started with
        $ python -m visdom.server
        and you probably want to run it from screen or tmux.
    '''

    @property
    def viz(self):
        return self._viz

    def __init__(self, fields=None, win=None, env=None, opts={}, port=8097, server="localhost"):
        super(BaseVisdomLogger, self).__init__(fields)
        self.win = win
        self.env = env
        self.opts = opts
        self._viz = visdom.Visdom(server="http://" + server, port=port)

    def log(self, *args, **kwargs):
        raise NotImplementedError(
            "log not implemented for BaseVisdomLogger, which is an abstract class.")

    def _viz_prototype(self, vis_fn):
        ''' Outputs a function which will log the arguments to Visdom in an appropriate way.

            Args:
                vis_fn: A function, such as self.vis.image
        '''
        def _viz_logger(*args, **kwargs):
            self.win = vis_fn(*args,
                              win=self.win,
                              env=self.env,
                              opts=self.opts,
                              **kwargs)
        return _viz_logger

    def log_state(self, state):
        """ Gathers the stats from self.trainer.stats and passes them into
            self.log, as a list """
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, state
            for f in field:
                parent, stat = stat, stat[f]
            results.append(stat)
        self.log(*results)


class VisdomSaver(object):
    ''' Serialize the state of the Visdom server to disk.
        Unless you have a fancy schedule, where different are saved with different frequencies,
        you probably only need one of these.
    '''

    def __init__(self, envs=None, port=8097, server="localhost"):
        super(VisdomSaver, self).__init__()
        self.envs = envs
        self.viz = visdom.Visdom(server="http://" + server, port=port)

    def save(self, *args, **kwargs):
        self.viz.save(self.envs)


class VisdomLogger(BaseVisdomLogger):
    '''
        A generic Visdom class that works with the majority of Visdom plot types.
    '''

    def __init__(self, plot_type, fields=None, win=None, env=None, opts={}, port=8097, server="localhost"):
        '''
            Args:
                fields: Currently unused
                plot_type: The name of the plot type, in Visdom

            Examples:
                >>> # Image example
                >>> img_to_use = skimage.data.coffee().swapaxes(0,2).swapaxes(1,2)
                >>> image_logger = VisdomLogger('image')
                >>> image_logger.log(img_to_use)

                >>> # Histogram example
                >>> hist_data = np.random.rand(10000)
                >>> hist_logger = VisdomLogger('histogram', , opts=dict(title='Random!', numbins=20))
                >>> hist_logger.log(hist_data)
        '''
        super(VisdomLogger, self).__init__(fields, win, env, opts, port, server)
        self.plot_type = plot_type
        self.chart = getattr(self.viz, plot_type)
        self.viz_logger = self._viz_prototype(self.chart)

    def log(self, *args, **kwargs):
        self.viz_logger(*args, **kwargs)




class VisdomFeatureMapsLogger(BaseVisdomLogger):
    '''
        A generic Visdom class that works with the majority of Visdom plot types.
    '''

    def __init__(self, plot_type, pad_value=1, nrow=2, fields=None, win=None, env=None, opts={}, port=8097, server="localhost"):
        '''
            Args:
                fields: Currently unused
                plot_type: The name of the plot type, in Visdom


        '''
        super(VisdomFeatureMapsLogger, self).__init__(fields, win, env, opts, port, server)
        self.plot_type = plot_type
        self.pad_value = pad_value
        self.nrow = nrow
        self.chart = getattr(self.viz, plot_type)
        self.viz_logger = self._viz_prototype(self.chart)

    def log(self, *args, **kwargs):
        self.viz_logger(*args, **kwargs)

    def images(self, bchw_tensor):
        self._viz.images(bchw_tensor, padding=self.pad_value, nrow=self.nrow, win=self.win,
        env=self.env,
        opts=self.opts)

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self._viz.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )
    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        Turning a batch of images to a grid
        e.g. input shape: (36, 64, 64)
        Will be a grid of 6x6, each grid is
        an image size 64x64
        """
        self.img('key', tv.utils.make_grid(
                    input_3d.cpu().unsqueeze(1), pad_value=self.pad_value, nrow=self.nrow))




class VisdomPlotLogger(BaseVisdomLogger):

    def __init__(self, plot_type, fields=None, win=None, env=None, opts={}, port=8097, server="localhost", name=None):
        '''
            Multiple lines can be added to the same plot with the "name" attribute (see example)
            Args:
                fields: Currently unused
                plot_type: {scatter, line}

            Examples:
                >>> scatter_logger = VisdomPlotLogger('line')
                >>> scatter_logger.log(stats['epoch'], loss_meter.value()[0], name="train")
                >>> scatter_logger.log(stats['epoch'], loss_meter.value()[0], name="test")
        '''
        super(VisdomPlotLogger, self).__init__(fields, win, env, opts, port, server)
        valid_plot_types = {
            "scatter": self.viz.scatter,
            "line": self.viz.line}
        self.plot_type = plot_type
        # Set chart type
        if plot_type not in valid_plot_types.keys():
            raise ValueError("plot_type \'{}\' not found. Must be one of {}".format(
                plot_type, valid_plot_types.keys()))
        self.chart = valid_plot_types[plot_type]

    def log(self, *args, **kwargs):
        if self.win is not None and self.viz.win_exists(win=self.win, env=self.env):
            if len(args) != 2:
                raise ValueError("When logging to {}, must pass in x and y values (and optionally z).".format(
                    type(self)))
            x, y = args
            self.chart(
                X=np.array([x]),
                Y=np.array([y]),
                update='append',
                win=self.win,
                env=self.env,
                opts=self.opts,
                **kwargs)
        else:
            if self.plot_type == 'scatter':
                chart_args = {'X': np.array([args])}
            else:
                chart_args = {'X': np.array([args[0]]),
                              'Y': np.array([args[1]])}
            self.win = self.chart(
                win=self.win,
                env=self.env,
                opts=self.opts,
                **chart_args)
            # For some reason, the first point is a different trace. So for now
            # we can just add the point again, this time on the correct curve.
            self.log(*args, **kwargs)


class VisdomTextLogger(BaseVisdomLogger):
    '''Creates a text window in visdom and logs output to it.

    The output can be formatted with fancy HTML, and it new output can
    be set to 'append' or 'replace' mode.

    Args:
        fields: Currently not used
        update_type: One of {'REPLACE', 'APPEND'}. Default 'REPLACE'.

    For examples, make sure that your visdom server is running.

    Example:
        >>> notes_logger = VisdomTextLogger(update_type='APPEND')
        >>> for i in range(10):
        >>>     notes_logger.log("Printing: {} of {}".format(i+1, 10))
        # results will be in Visdom environment (default: http://localhost:8097)

    '''
    valid_update_types = ['REPLACE', 'APPEND']

    def __init__(self, fields=None, win=None, env=None, opts={}, update_type=valid_update_types[0],
                 port=8097, server="localhost"):

        super(VisdomTextLogger, self).__init__(fields, win, env, opts, port, server)
        self.text = ''

        if update_type not in self.valid_update_types:
            raise ValueError("update type '{}' not found. Must be one of {}".format(
                update_type, self.valid_update_types))
        self.update_type = update_type

        self.viz_logger = self._viz_prototype(self.viz.text)

    def log(self, msg, *args, **kwargs):
        text = msg
        if self.update_type == 'APPEND' and self.text:
            self.text = "<br>".join([self.text, text])
        else:
            self.text = text
        self.viz_logger([self.text])

    def _log_all(self, stats, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        if prefix is not None:
            self.log(prefix)
        self.log(output)
        if suffix is not None:
            self.log(suffix)

    def _align_output(self, field_idx, output):
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _join_results(self, results):
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        output = []
        name = ''
        if isinstance(stat, dict):
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output




if __name__ == '__main__':
    visualize('/data/pun/results/features', '/data/pun/results/outputs')
