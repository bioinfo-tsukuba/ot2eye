#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from os import sep
import subprocess
import argparse
import csv
import yaml
import cv2
import shutil
from glob import glob
from scripts.obj_rec_eval import Obj_Rec_Eval
from scripts.plot import Plot


class OT2Eye():
	def __init__(self, img_dir, out_dir, model_labware, model_tip, threshold, train_yaml, answer_label_file, threshold_overlap, threshold_tip):
		#
		# argument
		#
		# img_dir: 推論画像ディレクトリ
		# out_dir: 結果保存ディレクトリ
		# model_labware: ラボウェアの検出モデル
		# model_tip:  チップの検出モデル
		# threshold:  検出閾値
		# train_yaml: 訓練データに用いたyamlファイル
		# answer_label_file: 解答ラベルファイル
		#
		# file name
		#
		DIR_TMP = "tmp" #一時出力ディレクトリ
		DIR_TMP_IMG_RESIZE     = DIR_TMP+sep+"images_resize" #縮小画像保存ディレクトリ
		DIR_TMP_IMG_TRIM       = DIR_TMP+sep+"images_trim"   #チップラックトリミング画像ディレクトリ
		DIR_TMP_DETECT_LABWARE = DIR_TMP+sep+"detect_labware"#ラボウェア検出結果ディレクトリ
		DIR_TMP_DETECT_TIP     = DIR_TMP+sep+"detect_tip"    #チップ検出結果ディレクトリ
		DIR_OUT_LABWARE_IMG = "images_labware"
		DIR_OUT_LABWARE_LBL = "labels_labware"
		DIR_OUT_TIP_IMG	    = "images_tip"
		DIR_OUT_TIP_LBL     = "labels_tip"
		DIR_OUT_MERGED_IMG  = "images_merged"
		DIR_OUT_MERGED_LBL  = "labels_merged"
		DIR_OUT_EVAL_IMG    = "images_evaluation"
		#
		# constants
		#
		self.SEPARATOR =" "  # labelファイルのセパレータ
		WIDTH_SMALL    = 640 # 縮小後画像幅
		HEIGHT_SMALL   = 480 # 縮小後画像高
		TIP_RACK_LABEL_NAME = "tip_rack" #チップラックのラベル名
		#
		# variables
		#
		self.out_dir = None # 出力ディレクトリ名
		self.width_original = 0
		self.height_original = 0
		self.plot_util = Plot()
		yaml_arr = None



		#
		# ディレクトリ処理
		#
		# 入力画像ディレクトリ存在確認
		if not os.path.isdir(img_dir):
			print("No such directory \"{}\"".format(img_dir))
			return
		# 保存先ディレクトリ生成
		out_dir      = self.make_output_dir(out_dir)
		self.out_dir = out_dir
		os.mkdir(out_dir+sep+DIR_TMP)


		#
		# 画像の縮小＆保存
		#
		print("##########################")
		print("# generate resize images #")
		print("##########################")
		os.mkdir(out_dir+sep+DIR_TMP_IMG_RESIZE)
		self.resize_save_img(img_dir, out_dir+sep+DIR_TMP_IMG_RESIZE, WIDTH_SMALL)


		#
		# リサイズ画像からラボウェア検出
		#
		print("##################")
		print("# detect labware #")
		print("##################")
		subprocess.run(["python3", "yolov5"+sep+"detect.py",\
				"--source", out_dir+sep+DIR_TMP_IMG_RESIZE, # 検出対象画像ディレクトリ
				"--project", out_dir, # 検出結果出力先ディレクトリ
				"--name", DIR_TMP_DETECT_LABWARE, # 検出結果ディレクトリ名
				"--weights", model_labware, # 検出モデル
				"--conf", str(threshold), # 検出閾値
				"--save-txt", # 推論ラベル出力
				"--save-conf", # 推論結果確率出力
				"--exist-ok"]) # 検出結果上書き

		#
		# 訓練用yamlfile読み込み
		#
		with open(train_yaml, 'r') as yamlfile:
			yaml_arr = yaml.safe_load(yamlfile)


		#
		# 同一ラボウェアの検出bboxが被りすぎていたら，確率が高い方のみ残す
		#
		print("###########################")
		print("# remove overlapping bbox #")
		print("###########################")
		self.remove_overlapping_bbox(out_dir+sep+DIR_TMP_DETECT_LABWARE+sep+"labels", threshold_overlap, yaml_arr, TIP_RACK_LABEL_NAME)
		

		#
		# 検出結果からトリミング画像生成
		#
		print("###########################")
		print("# generate triming images #")
		print("###########################")
		os.mkdir(out_dir+sep+DIR_TMP_IMG_TRIM)
		self.trim_tip_rack_img(\
				img_dir, # トリミング前画像ディレクトリ
				out_dir+sep+DIR_TMP_DETECT_LABWARE+sep+"labels", # ラボウェア検出結果ラベルディレクトリ
				yaml_arr, # 学習時yamlデータ配列
				out_dir+sep+DIR_TMP_IMG_TRIM, # トリミング結果画像ディレクトリ
				TIP_RACK_LABEL_NAME, # チップラックのラベル名
				WIDTH_SMALL, # トリミング結果画像幅
				HEIGHT_SMALL) # トリミング結果画像高


		#
		# トリミング画像からチップ検出
		#
		print("##############")
		print("# detect tip #")
		print("##############")
		if len(os.listdir(out_dir+sep+DIR_TMP_IMG_TRIM)) > 0:
			subprocess.run(["python3", "yolov5"+sep+"detect.py",\
					"--source", out_dir+sep+DIR_TMP_IMG_TRIM, # 検出対象画像ディレクトリ
					"--project", out_dir, # 検出結果出力先ディレクトリ
					"--name", DIR_TMP_DETECT_TIP, # 検出結果ディレクトリ名
					"--weights", model_tip, # 検出モデル
					"--conf", str(threshold), # 検出閾値
					"--save-txt", # 推論結果ラベル出力
					"--save-conf", # 推論結果確率出力
					"--exist-ok"])# 検出結果上書き
		else:
			print("    no tip rack")



		#
		# 同一チップの検出bboxが被りすぎていたら，確率が高い方のみ残す
		#
		print("###########################")
		print("# remove overlapping bbox #")
		print("###########################")
		self.remove_overlapping_bbox(out_dir+sep+DIR_TMP_DETECT_TIP+sep+"labels", threshold_overlap, yaml_arr, TIP_RACK_LABEL_NAME)


		#
		# 総合検出結果出力
		#
		print("################")
		print("# result merge #")
		print("################")
		# サブ出力ディレクトリ生成
		os.mkdir(out_dir+sep+DIR_OUT_LABWARE_IMG)
		os.mkdir(out_dir+sep+DIR_OUT_MERGED_IMG)

		# 統合ラベル生成
		self.make_merge_label(\
				out_dir+sep+DIR_OUT_MERGED_LBL, #出力先ディレクトリ
				out_dir+sep+DIR_TMP_DETECT_LABWARE+sep+"labels"+sep, # ラボウェアラベルディレクトリ
				out_dir+sep+DIR_TMP_DETECT_TIP+sep+"labels"+sep, # チップラベルディレクトリ
				yaml_arr, # 学習時yamlデータ配列
				TIP_RACK_LABEL_NAME, # チップラックのラベル名
				WIDTH_SMALL, HEIGHT_SMALL, # 縮小後画像サイズ
				threshold_tip) # チップがチップラックの外側にあると判定する面積の割合の閾値

		#結果ラベルコピー
		shutil.copytree(out_dir+sep+DIR_TMP_DETECT_LABWARE+sep+"labels", out_dir+sep+DIR_OUT_LABWARE_LBL)
		if len(os.listdir(out_dir+sep+DIR_TMP_IMG_TRIM)) > 0:
			os.mkdir(out_dir+sep+DIR_OUT_TIP_IMG)
			shutil.copytree(out_dir+sep+DIR_TMP_DETECT_TIP+sep+"labels", out_dir+sep+DIR_OUT_TIP_LBL)


		#
		# 同一ラボウェアチップの検出bboxが被りすぎていたら，確率が高い方のみ残す
		#
		print("###########################")
		print("# remove overlapping bbox #")
		print("###########################")
		self.remove_overlapping_bbox(out_dir+sep+DIR_OUT_MERGED_LBL, threshold_overlap, yaml_arr, TIP_RACK_LABEL_NAME)


		#ラボウェア＆チップ検出結果画像静止絵
		self.make_bbox_image(img_dir, out_dir+sep+DIR_OUT_MERGED_LBL,
				out_dir+sep+DIR_OUT_MERGED_IMG, yaml_arr)
		#ラボウェア検出結果画像生成
		self.make_bbox_image(img_dir, out_dir+sep+DIR_TMP_DETECT_LABWARE+sep+"labels"+sep,
				out_dir+sep+DIR_OUT_LABWARE_IMG, yaml_arr)
		#チップ検出結果画像生成
		self.make_bbox_image(out_dir+sep+DIR_TMP_IMG_TRIM, out_dir+sep+DIR_TMP_DETECT_TIP+sep+"labels"+sep,
				out_dir+sep+DIR_OUT_TIP_IMG, "tip")

		#
		# 総合検出結果出力
		#
		print("#######################")
		print("# Position_Inference  #")
		print("#######################")
		self.position_inference(img_dir, out_dir+sep+DIR_OUT_LABWARE_LBL, yaml_arr, out_dir)
		



		if answer_label_file != None:
			print("##############")
			print("# evaluation #")
			print("##############")
			#正解ラベル画像出力
			os.mkdir(out_dir+sep+DIR_OUT_EVAL_IMG)
			# def make_bbox_image(self, ori_img_dir, label_dir, save_dir, yaml_arr, ans=False):
			self.make_bbox_image(out_dir+sep+DIR_OUT_MERGED_IMG, answer_label_file, out_dir+sep+DIR_OUT_EVAL_IMG, yaml_arr, True)






	#
	# ラボウェア＆チップのラベル統合
	#
	def make_merge_label(self, out_dir, dir_labware_lbl, dir_tip_lbl, yaml_arr, TIP_RACK_LABEL_NAME, WIDTH_SMALL, HEIGHT_SMALL, threshold_tip):
		TIP_LABEL_NUM  = 0 # チップのラベル番号
		RACK_LABEL_NUM = 0 # チップラックのラベル番号

		# 出力ディレクトリ生成
		os.mkdir(out_dir)

		# チップのラベル番号取得
		TIP_LABEL_NUM  = yaml_arr["nc"]
		RACK_LABEL_NUM = yaml_arr["names"].index(TIP_RACK_LABEL_NAME)


		# ラボウェアラベルファイルループ
		for fname_labware_label in os.listdir(dir_labware_lbl):
			base_name_labware_label = fname_labware_label.rsplit(".",1)[0]

			# 統合ラベルファイル生成
			merge_label_file = open(out_dir+sep+fname_labware_label, "a")			

			# ラボウェアラベルファイルopen
			with open(dir_labware_lbl+sep+fname_labware_label, 'r') as txtfile_labware:
				num_rack = 0 #画像内のチップラックの数
				reader = csv.reader(txtfile_labware, delimiter=self.SEPARATOR) #ファイル読み込み

				# ラベルファイルの中身の行ループ
				for row in reader:
					merge_label_file.write(" ".join(row)+"\n") #読み込んだ行をそのまま書き込み
					
					# チップラックの場合処理
					if row[0] == str(RACK_LABEL_NUM):
						fname_tip_label = base_name_labware_label+"_"+str(num_rack)+".txt"

						# チップラベルファイルが存在すれば
						if os.path.isfile(dir_tip_lbl+sep+fname_tip_label):
							# チップラベル情報取得
							tip_label_arr = self.label_file_to_arr(dir_tip_lbl+sep+fname_tip_label)

							# ラベルファイルの中身の行のループ
							for i in range(len(tip_label_arr)):
								# オリジナル座標系におけるチップラックのbbox
								rack_pos_w  = float(row[1])
								rack_pos_h  = float(row[2])
								rack_size_w = float(row[3])
								rack_size_h = float(row[4])
								rack_size_w_pre = float(row[3])
								rack_size_h_pre = float(row[4])
								# チップラック座標におけるチップのbbox
								tip_pos_w  = float(tip_label_arr[i][1])
								tip_pos_h  = float(tip_label_arr[i][2])
								tip_size_w = float(tip_label_arr[i][3])
								tip_size_h = float(tip_label_arr[i][4])

								# チップラックのトリミングサイズの計算
								rack_aspect = (rack_size_h_pre*self.height_original) / float(rack_size_w_pre*self.width_original)
								if rack_aspect <= (HEIGHT_SMALL / float(WIDTH_SMALL)): #横長
									resize_rate = float(WIDTH_SMALL) / float(self.width_original*rack_size_w_pre)
									rack_size_h = (HEIGHT_SMALL / float(resize_rate)) / float(self.height_original)
								else:
									resize_rate = float(HEIGHT_SMALL) / float(self.height_original*rack_size_h_pre)
									rack_size_w = (WIDTH_SMALL / float(resize_rate)) / float(self.width_original)
								# チップラベルをオリジナル画像の座標系へ変換
								tip_pos_w_ori  = rack_pos_w-0.5*rack_size_w+tip_pos_w*rack_size_w
								tip_pos_h_ori  = rack_pos_h-0.5*rack_size_h+tip_pos_h*rack_size_h
								tip_size_w_ori = rack_size_w*tip_size_w
								tip_size_h_ori = rack_size_h*tip_size_h
								# if not ((tip_pos_w_ori+0.5*tip_size_w_ori < rack_pos_w-0.5*rack_size_w_pre) \
								# 		or (tip_pos_w_ori-0.5*tip_size_w_ori > rack_pos_w+0.5*rack_size_w_pre) \
								# 		or (tip_pos_h_ori+0.5*tip_size_h_ori < rack_pos_h-0.5*rack_size_h_pre) \
								# 		or (tip_pos_h_ori-0.5*tip_size_h_ori > rack_pos_h+0.5*rack_size_h_pre)):
								# チップの内一定割合の面積がチップラックbboxと被っていなければ
								if self.calc_overlapping_area(list(map(float,row)), \
										(0,tip_pos_w_ori,tip_pos_h_ori,tip_size_w_ori,tip_size_h_ori)) \
										> threshold_tip*tip_size_w_ori*tip_size_h_ori:
									tip_label_arr[i][0] = str(TIP_LABEL_NUM)
									tip_label_arr[i][1] = str(round(tip_pos_w_ori, 6))
									tip_label_arr[i][2] = str(round(tip_pos_h_ori, 6))
									tip_label_arr[i][3] = str(round(tip_size_w_ori, 6))
									tip_label_arr[i][4] = str(round(tip_size_h_ori, 6))
									# チップラベル情報書き込み
									merge_label_file.write(" ".join(tip_label_arr[i])+"\n")

						num_rack += 1

			# 統合ラベルファイルclose
			merge_label_file.close()


	#
	# 重複bboxの削除
	# 
	def remove_overlapping_bbox(self, dir_label, threshold, yaml_arr, TIP_RACK_LABEL_NAME):
		TIP_LABEL_NUM  = yaml_arr["nc"]
		RACK_LABEL_NUM = yaml_arr["names"].index(TIP_RACK_LABEL_NAME)

		# ラベルファイルループ
		for label_file_name in os.listdir(dir_label):
			remove_idx = []
			label_arr = self.label_file_to_arr(dir_label+sep+label_file_name)

			# 重複ラベルの検出
			for idx1 in range(len(label_arr)):
				for idx2 in range(idx1+1, len(label_arr)):
					box1 = list(map(float, label_arr[idx1]))
					box2 = list(map(float, label_arr[idx2]))

					overlapping_area  = self.calc_overlapping_area(box1, box2) # 重複部分面積
					smaller_bbox_area = min(box1[3]*box1[4], box2[3]*box2[4]) # 小さい方のbboxの面積

					if overlapping_area >= threshold * smaller_bbox_area: # 重複部分が大きければ
						if not ((box1[0] == TIP_LABEL_NUM and box2[0] == RACK_LABEL_NUM) or (box1[0] == RACK_LABEL_NUM and box2[0] == TIP_LABEL_NUM)): # チップとチップラックの重複は許容
							if box1[5] < box2[5]: # 確率が低い方は削除
								remove_idx.append(idx1)
							else:
								remove_idx.append(idx2)
			
			# 重複ラベル削除
			if len(remove_idx) > 0:
				# print(label_arr)
				# print(remove_idx)
				remove_idx.sort(reverse=True)
				# for i in range(len(remove_idx)):
				# 	del label_arr[remove_idx[i]-i]
				# print(label_arr)
				# print(label_file_name)
				# print(len(remove_idx))
				# # 重複ラベルを配列から削除
				for i in remove_idx:
					del label_arr[i]

				# ラベルファイルの書き換え
				with open(dir_label+sep+label_file_name, 'w', newline='') as txtfile:
					writer = csv.writer(txtfile, delimiter=' ')
					for label in label_arr:
						writer.writerow(label)

	# bboxの重複部分の面積計算
	def calc_overlapping_area(self, row1, row2):
		x1 = max(row1[1] - row1[3] / 2, row2[1] - row2[3] / 2)
		x2 = min(row1[1] + row1[3] / 2, row2[1] + row2[3] / 2)
		y1 = max(row1[2] - row1[4] / 2, row2[2] - row2[4] / 2)
		y2 = min(row1[2] + row1[4] / 2, row2[2] + row2[4] / 2)

		return max(0, x2-x1) * max(0, y2-y1)


	#
	# 検出データからbbox付き画像生成
	#
	def make_bbox_image(self, ori_img_dir, label_dir, save_dir, yaml_arr, eval_mode=False):
		object_name = None # 検出オブジェクト名配列

		# 訓練yaml取得(ラベルデータ取得)
		if yaml_arr == "tip":
			object_name = ["tip"]
		else:
			object_name = yaml_arr["names"].copy()

		# 元画像ループ
		for img_name in os.listdir(ori_img_dir):
			base_name = img_name.rsplit(".",1)[0]
		
			# ラベルの有無確認
			if not os.path.isfile(label_dir+sep+base_name+".txt"): # ラベルがなければスキップ
				continue
			else: # ラベルがあれば以下の処理
				# 画像取得
				img = cv2.imread(ori_img_dir+sep+img_name)
				# ラベルループ
				for label_row in self.label_file_to_arr(label_dir+sep+base_name+".txt"):
					self.plot_util.label_row_to_bbox(img, label_row, object_name, eval_mode)
				# 保存
				cv2.imwrite(save_dir+sep+img_name, img)



	#
	# 保存先rootディレクトリ生成
	#
	def make_output_dir(self, out_dir):
		MAX_SEQ_NUM = 1024 # 最大連番数

		if not os.path.isdir(out_dir): #存在しなければ生成
			os.mkdir(out_dir)
			print("The directory \"{}\" was made.".format(out_dir))
		else: #存在すれば連番で代わりを生成
			while out_dir[-1]==sep:
				out_dir = out_dir[:-1] # ディレクトリパスの最後に"/"があれば消す
			# 連番ディレクトリ名存在確認
			for i in range(1,MAX_SEQ_NUM):
				out_dir_cdd = out_dir + str(i)
				if not os.path.isdir(out_dir_cdd):
					os.mkdir(out_dir_cdd)
					print("The directory \"{}\" already exists. \"{}\" was made instead.".format(out_dir, out_dir_cdd))
					out_dir = out_dir_cdd
					break

		# ディレクトリ名戻し
		return out_dir



	#
	# オリジナル画像からチップラックをトリミング
	#
	def trim_tip_rack_img(self, origin_dir_path, label_dir_path, yaml_arr, output_path,
			tip_rack_label="tip_rack", trimmed_img_w=640.0, trimmed_img_h=480.0):

		# loop for all file in original image directory
		for img_name in os.listdir(origin_dir_path):
			base_name    = img_name.rsplit(".",1)[0]
			img_ext_name = img_name.rsplit(".",1)[1]
			label_name   = base_name+".txt"

			# ラベルがなければ飛ばす
			if not os.path.isfile(label_dir_path+sep+label_name):
				continue

			# get image info
			ori_img = cv2.imread(origin_dir_path+sep+img_name)
			ori_height, ori_width = ori_img.shape[:2]
			# get label arr
			labels = self.label_file_to_arr(label_dir_path+sep+label_name)

			# get tip rack coordinate
			index = 0
			for row in labels: # loop for all row in detected label text file
				if int(row[0]) == yaml_arr["names"].index(tip_rack_label): # if this row is about tip rack 
					# get detected object info
					obj_pos_w  = ori_width *float(row[1])
					obj_pos_h  = ori_height*float(row[2])
					obj_size_w = ori_width *float(row[3])
					obj_size_h = ori_height*float(row[4])

					# trimming from original image
					rate = 0
					if (obj_size_h / obj_size_w) <= (trimmed_img_h / trimmed_img_w): #横長
						rate =  trimmed_img_w / obj_size_w
					else:
						rate =  trimmed_img_h / obj_size_h
					trim_img = cv2.resize(ori_img, dsize=None, fx=rate, fy=rate)
					trim_img = trim_img[\
							round(obj_pos_h*rate - trimmed_img_h*0.5):\
							round(obj_pos_h*rate + trimmed_img_h*0.5),\
							round(obj_pos_w*rate - trimmed_img_w*0.5):\
							round(obj_pos_w*rate + trimmed_img_w*0.5)]

					# output trimmed image
					out_file_name = base_name + "_{}".format(index) + "." + img_ext_name
					cv2.imwrite(output_path+sep+out_file_name, trim_img)

					index += 1

	

	#
	# ラベルファイルから配列へ変換
	#
	def label_file_to_arr(self, file_path):
		labels = None
		with open(file_path, 'r') as txtfile:
			reader = csv.reader(txtfile, delimiter=self.SEPARATOR)
			labels = [row for row in reader]

		return labels



	#
	# 画像の縮小＆保存
	#
	def resize_save_img(self, img_dir, out_dir, w_aft):
		for img_name in os.listdir(img_dir):
			# read image
			img = cv2.imread(img_dir+sep+img_name)
			# get original img size
			self.height_original, self.width_original = img.shape[:2]
			# resize
			h_bef, w_bef = img.shape[:2]
			h_aft = round(h_bef * (w_aft / w_bef))
			img = cv2.resize(img, dsize=(w_aft, h_aft))
			cv2.imwrite(out_dir+sep+img_name, img)

	
	#
	# 位置推論
	#
	def position_inference(self, img_dir_path, label_dir_path, yaml_arr, out_file_path):
		object_name = yaml_arr["names"].copy()
	
		result_arr = []
		result_arr.append(["Image_file", "Deck", "Labware"\
			, "Confidence", "x_pos", "y_pos", "width", "height"])

		# loop for label files
		for img_file_name in  sorted(os.listdir(img_dir_path)):
			base_name        = img_file_name.rsplit(".",1)[0]
			label_file_name  = base_name+".txt"

			# get label arr
			labels = self.label_file_to_arr(label_dir_path+sep+label_file_name)

			# sort for position inference
			labels.sort(key=lambda x:x[2], reverse=True)
			labels[0:3]  = sorted(labels[0:3],  key=lambda x:x[1])
			labels[3:6]  = sorted(labels[3:6],  key=lambda x:x[1])
			labels[6:9]  = sorted(labels[6:9],  key=lambda x:x[1])
			labels[9:11] = sorted(labels[9:11], key=lambda x:x[1])

			# for i in range(len(labels)):
			# 	print("{:>15} {} {:.04f} {:.04f}".format(object_name[int(labels[i][0])]
			# 		, labels[i][0], float(labels[i][1]), float(labels[i][2])))
			# print("")

			for i in range(len(labels)):
				result_arr.append([img_file_name, i+1, object_name[int(labels[i][0])],\
						"{:.04f}".format(float(labels[i][5])),\
						"{:.04f}".format(float(labels[i][1])),\
						"{:.04f}".format(float(labels[i][2])),\
						"{:.04f}".format(float(labels[i][3])),\
						"{:.04f}".format(float(labels[i][4]))])

		# output evalutate file
		with open(out_file_path+sep+"position_inference.csv", "w") as out_file:
			writer = csv.writer(out_file, delimiter="\t")
			writer.writerows(result_arr)




if __name__ == '__main__':
	#
	# argument
	#
	prs = argparse.ArgumentParser()
	# image dir to detect
	prs.add_argument("image_dir", type=str,
			help="directory path of image files.")
	# output dir path
	prs.add_argument("--out-dir", type=str, required=False, default="out",
			help="directory path of output files.")
	# labware detection model
	prs.add_argument("--model-labware", type=str, required=False,
			default="."+sep+"model"+sep+"exp_20230622+20230525"+sep+"weights"+sep+"best.pt",
			help="file path of labware detection model.")
	# tip detection model
	prs.add_argument("--model-tip", type=str, required=False,
			default="."+sep+"model"+sep+"exp_20231127+20231026+20220718+20220624_trim"+sep+"weights"+sep+"best.pt",
			help="file path of labware detection model.")
	# threshold of confidence
	prs.add_argument("--threshold-conf", type=float, required=False,
			default=0.7,
			help="confidence threshold of detection. (default: 0.7)")
	# yaml file of training labware
	prs.add_argument("--labware-train-yaml", type=str, required=False,
			default="."+sep+"model"+sep+"dataset_20230622+20230525.yaml",
			help="yaml file path of training labware.")
	# evaluation mode
	prs.add_argument("--evaluate", type=str, required=False, default=None,
			help="directory path of answer labels file.")
	# threshold of area for determining that the same object is detected
	prs.add_argument("--threshold-overlap", type=float, required=False,
			default=0.5,
			help="threshold of area for determining that the same object is detected. (default: 0.5)")
	# threshold for determining tips outside the tiprack
	prs.add_argument("--threshold-tip", type=float, required=False,
			default=0.5,
			help="threshold for determining tips outside the tiprack. (default: 0.5)")
	args = prs.parse_args()

	#
	# detection
	#
	ot2eye = OT2Eye(args.image_dir, args.out_dir, args.model_labware, args.model_tip, args.threshold_conf, args.labware_train_yaml, args.evaluate, args.threshold_overlap, args.threshold_tip)

	#
	# evaluation mode
	#
	if args.evaluate != None:
		obj_rec_eval = Obj_Rec_Eval(\
				ot2eye.out_dir,\
				ot2eye.out_dir+sep+"labels_merged",\
				ot2eye.out_dir+sep+"images_merged",\
				args.evaluate)

	# example
	# python3 ot2eye.py dataset/20220718_large/
