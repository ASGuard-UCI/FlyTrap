# model settings
model = dict(type='mixformer_cvt_online_scores',
             checkpoint='./models/MixFormer/models/mixformer_online_22k.pth.tar')

# tracker settings, used for evaluation
tracker = dict(type='mixformer_cvt_online_scores_tracker',
               params=dict(
                   model='mixformer_online_22k.pth.tar',
                   update_interval=10000, online_sizes=1, search_area_scale=4.5,
                   max_score_decay=1.0, vis_attn=0))