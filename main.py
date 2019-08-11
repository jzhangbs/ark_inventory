import cv2
import numpy as np
from scipy.stats import mode

import number_recog  # must import first, or tesseract reports error 127
import region_proposal
import item_classify
import get_scene


if __name__ == '__main__':

    inventory = {}
    while True:

        roi_boxes = []
        while (len(roi_boxes) == 0):
            print('get scene')
            # scene = get_scene.from_file('c:/users/zhang/Desktop/ark_scene.png')
            scene = get_scene.from_adb()
            assert scene is not None

            wh_ratio = scene.shape[1] / scene.shape[0]
            scene = cv2.resize(scene, (int(720 * wh_ratio), 720))

            print('region proposal')
            roi_boxes = region_proposal.region_proposal(scene)

        roi_list = []
        roi_list_copy = []
        for x, y, box_size in roi_boxes:
            if y < 0 or x < 0 or y + box_size >= scene.shape[0] or x + box_size >= scene.shape[1]:
                continue
            roi = scene[y:y + box_size, x:x + box_size, :]
            roi = cv2.resize(roi, (128, 128))
            roi_copy = roi.copy()
            roi_list_copy.append(roi_copy)
            roi = roi / 255 * 2 - 1
            roi = np.transpose(roi, (2, 0, 1))
            roi_list.append(roi)

        print('item classify')
        predicts, probs = item_classify.predict(roi_list)

        print('number recognition')
        # with mp.Pool(processes=mp.cpu_count()) as p:
        #     numbers = p.map(number_recog.number_recog, roi_list_copy)
        numbers = [number_recog.number_recog(roi) for roi in roi_list_copy]

        # partial_inventory = {predict: number for predict, number in zip(predicts, numbers) if predict != 'Not_Support'}
        # inventory.update(partial_inventory)
        for predict, number in zip(predicts, numbers):
            if predict == 'Not_Support':
                continue
            if inventory.get(predict) is None:
                inventory[predict] = [number]
            else:
                inventory[predict].append(number)

        inventory_final = {k: mode(v)[0][0] for k, v in inventory.items() if len(v) >= 3}
        print(inventory_final)

        to_graueneko = ['D32_Gang', 'Shuang_Ji_Na_Mi_Pian', 'Ju_He_Ji', 'RMA70-24', 'Wu_Shui_Yan_Mo_Shi', 'San_Shui_Meng_Kuang', 'Bai_Ma_Chun', 'Gai_Liang_Zhuang_Zhi', 'Tong_Zhen_Lie', 'Yi_Tie_Kuai', 'Ju_Suan_Zhi_Kuai', 'Tang_Ju_Kuai', 'Ti_Chun_Yuan_Yan', 'Quan_Xin_Zhuang_Zhi', 'Tong_Ning_Ji_Zu', 'Yi_Tie_Zu', 'Ju_Suan_Zhi_Zu', 'Tang_Zu', 'Gu_Yuan_Yan_Zu', 'Zhuang_Zhi', 'Tong_Ning_Ji', 'Yi_Tie', 'Ju_Suan_Zhi', 'Tang', 'Gu_Yuan_Yan', 'Po_Sun_Zhuang_Zhi', 'Shuang_Tong', 'Yi_Tie_Sui_Pian', 'Zhi_Yuan_Liao', 'Dai_Tang', 'Yuan_Yan', 'Niu_Zhuan_Chun', 'Yan_Mo_Shi', 'Qing_Meng_Kuang', 'RMA70-12']
        graueneko_uri = 'https://ak.graueneko.xyz/akevolve.html?n=0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0+0&h=' + '+'.join([str(inventory_final[name]) if inventory_final.get(name) is not None else '0' for name in to_graueneko]) + '&o=3+4+5'
        print(graueneko_uri)
