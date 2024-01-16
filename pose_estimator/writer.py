def state(depth_l1_loss_avg, pos_l1_loss_avg,orn_l1_loss_avg,depth_smooth_loss_avg, depth_ssim_loss_avg,loss_avg,learning_rate):
    return{
        'depth_l1_loss_avg':depth_l1_loss_avg,
        'pos_l1_loss_avg':pos_l1_loss_avg,
        'orn_l1_loss_avg': orn_l1_loss_avg,
        'depth_smooth_loss_avg':depth_smooth_loss_avg,
        'depth_ssim_loss_avg':depth_ssim_loss_avg,
        'loss_avg':loss_avg,
        'learning_rate':learning_rate
    }
def writeSummary(writer,stats,episode):
    for key in stats.keys():
        writer.add_scalar(tag='Info/{}'.format(key), scalar_value=stats[key], global_step=episode)
    writer.flush()