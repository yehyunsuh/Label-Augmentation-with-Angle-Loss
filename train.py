import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pprint

from tqdm import tqdm
from utility.log import log_results, log_terminal
from utility.train import set_parameters, rmse, geom_element, angle_element, dist_element
from utility.visualization import visualize

def train_function(args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, loss_fn_angle, loss_fn_dist, optimizer, train_loader):
    total_loss, total_pixel_loss, total_geom_loss, total_angle_loss = 0, 0, 0, 0
    total_num_noma, total_num_pred_noma, total_loss_dist = 0, 0, 0
    model.train()

    for image, label, _, _, label_list in tqdm(train_loader):
        image = image.to(device=DEVICE)
        label = label.float().to(device=DEVICE)
        
        # Pixel Loss
        prediction = model(image)
        loss_pixel = loss_fn_pixel(prediction, label)

        # Geometry Loss - Mean Pixel Distance
        predict_spatial_mean, label_spatial_mean = geom_element(torch.sigmoid(prediction), label)
        loss_geometry = loss_fn_geometry(predict_spatial_mean, label_spatial_mean)

        # Angle Loss
        predict_angle, label_angle = angle_element(args, prediction, label_list, DEVICE, 'train')
        loss_angle = loss_fn_angle(predict_angle, label_angle)

        # Distance Loss - Max Pixel Distance
        predict_max_pixel, label_pixel = dist_element(args, prediction, label_list, DEVICE)
        loss_dist = loss_fn_dist(predict_max_pixel, label_pixel)

        # ## NoMa Loss
        # num_noma, num_pred_noma = 0, 0
        # for i in range(len(label_list[0])): # 0~17
        #     for j in range(0,len(label_list),2):# 0~39
        #         if label_list[j][i].item() == 0 and label_list[j+1][i].item() == 0:
        #             num_noma += 1
        #             # print(i,j//2, end='\t')
        #             # print(torch.max(prediction_sigmoid[i][j//2]).item()) # 18, 20, 512, 512
        #             if torch.max(prediction_sigmoid[i][j//2]).item() > args.threshold:
        #                 num_pred_noma += 1

        # Total Loss
        # if args.geom_loss and args.angle_loss:
        #     loss = args.pixel_loss_weight * loss_pixel + args.geom_loss_weight * loss_geometry + args.angle_loss_weight * loss_angle
        # elif args.geom_loss and not args.angle_loss:
        #     loss = args.pixel_loss_weight * loss_pixel + args.geom_loss_weight * loss_geometry
        # elif not args.geom_loss and args.angle_loss:
        #     loss = args.pixel_loss_weight * loss_pixel + args.angle_loss_weight * loss_angle
        # else:
        #     loss = loss_pixel

        loss = args.pixel_loss_weight * loss_pixel #+ args.dist_loss_weight * loss_dist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss          += loss.item()
        total_pixel_loss    += loss_pixel.item() 
        total_geom_loss     += loss_geometry.item()
        total_angle_loss    += loss_angle.item()
        total_loss_dist     += loss_dist.item()
    #     total_num_noma      += num_noma
    #     total_num_pred_noma += num_pred_noma

    # if total_num_pred_noma != 0:
    #     total_noma_loss = total_num_noma/total_num_pred_noma
    # else:
    #     total_noma_loss = 0

    return total_loss, total_pixel_loss, total_geom_loss, total_angle_loss, total_loss_dist


def validate_function(args, DEVICE, model, epoch, val_loader):
    print("=====Starting Validation=====")
    model.eval()

    dice_score, rmse_total = 0, 0
    extracted_pixels_list = []
    rmse_list = [[0]*len(val_loader) for _ in range(args.output_channel)]
    angle_list = [[0]*len(val_loader) for _ in range(len(args.label_for_angle))]

    with torch.no_grad():
        for idx, (image, label, image_path, image_name, label_list) in enumerate(tqdm(val_loader)):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            image_path = image_path[0]
            image_name = image_name[0].split('.')[0]
            
            prediction = model(image)
            
            # validate angle difference
            predict_angle, label_angle = angle_element(args, prediction, label_list, DEVICE, 'validate')

            for i in range(len(args.label_for_angle)):
                angle_list[i][idx] = abs(label_angle[i] - predict_angle[i])

            # validate mean geom difference
            predict_spatial_mean, label_spatial_mean = geom_element(torch.sigmoid(prediction), label)

            ## get rmse difference
            rmse_list, index_list = rmse(args, prediction, label_list, idx, rmse_list)
            extracted_pixels_list.append(index_list)

            ## make predictions to be 0. or 1.
            prediction_binary = (prediction > 0.5).float()
            dice_score += (2 * (prediction_binary * label).sum()) / ((prediction_binary + label).sum() + 1e-8)

            ## visualize
            if epoch % args.dilation_epoch == 0 or epoch % args.dilation_epoch == (args.dilation_epoch-1):
                if not args.no_visualization:
                    visualize(
                        args, idx, image_path, image_name, label_list, epoch, extracted_pixels_list, prediction, prediction_binary,
                        predict_spatial_mean, label_spatial_mean, 'train'
                    )
            # elif epoch % (args.epochs // 10) == 0:
            #     visualize(
            #         args, idx, image_path, image_name, label_list, epoch, extracted_pixels_list, prediction, prediction_binary,
            #         predict_spatial_mean, label_spatial_mean, 'train'
            #     )

    dice = dice_score/len(val_loader)

    rmse_sum = 0
    for i in range(len(rmse_list)):
        for j in range(len(rmse_list[i])):
            if rmse_list[i][j] != -1:
                rmse_sum += rmse_list[i][j]

    rmse_mean = rmse_sum/(len(val_loader)*args.output_channel)
    print(f"Dice score: {dice}")
    print(f"Average Pixel to Pixel Distance: {rmse_mean}")

    angle_value = []
    for i in range(len(args.label_for_angle)):
        angle_value.append(sum(angle_list[i]))
    angle_value.append(sum(list(map(sum, angle_list))))

    return dice, rmse_mean, rmse_list, angle_value


def train(args, model, DEVICE):
    best_loss, best_rmse_mean, best_angle_diff = np.inf, np.inf, np.inf
    loss_fn_geometry, loss_fn_angle, loss_fn_dist = nn.MSELoss(), nn.MSELoss(), nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        print(f"\nRunning Epoch # {epoch}")
        
        if epoch % args.dilation_epoch == 0:
            args, loss_fn_pixel, train_loader, val_loader = set_parameters(
                args, model, epoch, DEVICE
            )

        loss, loss_pixel, loss_geom, loss_angle, loss_dist = train_function(
            args, DEVICE, model, loss_fn_pixel, loss_fn_geometry, loss_fn_angle, loss_fn_dist, optimizer, train_loader
        )
        dice, rmse_mean, rmse_list, angle_value = validate_function(
            args, DEVICE, model, epoch, val_loader
        )

        print("Average Train Loss: ", loss/len(train_loader))
        if best_loss > loss:
            print("=====New best model=====")
            best_loss = loss

        if best_rmse_mean > rmse_mean:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            }
            # torch.save(checkpoint, f'./results/{args.wandb_name}/best.pth')
            best_rmse_mean = rmse_mean
        
        if best_angle_diff > angle_value[len(args.label_for_angle)]:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            }
            # torch.save(checkpoint, f'./results/{args.wandb_name}/best.pth')
            best_angle_diff = angle_value[len(args.label_for_angle)]

        if args.wandb:
            log_results(
                loss, loss_pixel, loss_geom, loss_angle, loss_dist,
                dice, rmse_mean, best_rmse_mean, rmse_list, best_angle_diff, angle_value,
                len(train_loader), len(val_loader), len(args.label_for_angle)
            )
    log_terminal(args, rmse_list)