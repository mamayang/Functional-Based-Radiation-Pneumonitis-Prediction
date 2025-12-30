import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np


class ImagePairProcessor:
    def verify_alignment_with_different_origins(self, target_img, perf_img, img_type):
        """验证图像是否对齐（target_img可以是CT或Dose）"""
        # 获取图像信息
        target_origin = np.array(target_img.GetOrigin())
        perf_origin = np.array(perf_img.GetOrigin())
        target_direction = np.array(target_img.GetDirection())
        perf_direction = np.array(perf_img.GetDirection())
        target_spacing = np.array(target_img.GetSpacing())
        perf_spacing = np.array(perf_img.GetSpacing())

        print(f"\n{img_type}图像信息:")
        print(f"Origin: {target_origin}")
        print(f"Direction: {target_direction}")
        print(f"Spacing: {target_spacing}")
        print("\nPerfusion图像信息:")
        print(f"Origin: {perf_origin}")
        print(f"Direction: {perf_direction}")
        print(f"Spacing: {perf_spacing}")

        # 检查方向和间距是否匹配
        direction_match = np.allclose(target_direction, perf_direction, atol=self.tolerance)
        spacing_match = np.allclose(target_spacing, perf_spacing, atol=self.tolerance)

        # 计算origin差异
        origin_diff = target_origin - perf_origin
        print(f"\nOrigin差异: {origin_diff} mm")

        # 将origin差异转换为体素单位
        voxel_diff = origin_diff / target_spacing
        print(f"体素单位的差异: {voxel_diff} voxels")

        return {
            'direction_match': direction_match,
            'spacing_match': spacing_match,
            'origin_diff_mm': origin_diff,
            'origin_diff_voxels': voxel_diff,
            'is_aligned': direction_match and spacing_match
        }

    def find_matching_folders(self):
        """找到匹配的文件夹"""
        perf_folders = {f.name: f for f in self.perfusion_root.iterdir() if f.is_dir()}
        target_folders = {f.name: f for f in self.target_root.iterdir() if f.is_dir()}

        common_folders = set(perf_folders.keys()) & set(target_folders.keys())

        matches = []
        for folder_name in common_folders:
            perf_file = perf_folders[folder_name] / "moved_back_perfusion.nii"
            ct_file = target_folders[folder_name] / "ct.mha"
            dose_file = target_folders[folder_name] / "Dose.mha"

            if perf_file.exists():
                if ct_file.exists():
                    matches.append({
                        'folder_name': folder_name,
                        'perfusion_path': perf_file,
                        'target_path': ct_file,
                        'type': 'CT'
                    })
                if dose_file.exists():
                    matches.append({
                        'folder_name': folder_name,
                        'perfusion_path': perf_file,
                        'target_path': dose_file,
                        'type': 'Dose'
                    })

        print(f"找到 {len(matches)} 对匹配的文件")
        return matches

    def process_image_pair(self, perf_path, target_path, img_type, output_folder):
        """处理一对图像"""
        print(f"\n处理 {img_type} 图像...")

        # 读取图像
        perf_img = sitk.ReadImage(str(perf_path))
        target_img = sitk.ReadImage(str(target_path))

        # 验证对齐
        alignment_results = self.verify_alignment_with_different_origins(
            target_img, perf_img, img_type)

        if not alignment_results['is_aligned']:
            raise ValueError(f"{img_type}图像与Perfusion图像未正确对齐")

        # 获取bbox和裁剪信息
        crop_info = self.get_crop_info(perf_img, target_img, alignment_results)

        # 执行裁剪
        cropped_img = self.crop_image(target_img, crop_info)

        # 保存结果
        output_path = output_folder / f"cropped_{img_type.lower()}.mha"
        sitk.WriteImage(cropped_img, str(output_path))

        return cropped_img, crop_info, alignment_results

    def process_all_pairs(self):
        """处理所有匹配的图像对"""
        matches = self.find_matching_folders()
        results = []

        for match in matches:
            folder_name = match['folder_name']
            img_type = match['type']

            try:
                output_folder = self.output_root / folder_name
                output_folder.mkdir(parents=True, exist_ok=True)

                cropped_img, crop_info, alignment_results = self.process_image_pair(
                    match['perfusion_path'],
                    match['target_path'],
                    img_type,
                    output_folder
                )

                results.append({
                    'folder': folder_name,
                    'type': img_type,
                    'success': True,
                    'crop_info': crop_info,
                    'alignment_info': alignment_results
                })

                print(f"成功处理 {folder_name} 的 {img_type} 图像")

            except Exception as e:
                print(f"处理 {folder_name} 的 {img_type} 图像时出错: {str(e)}")
                results.append({
                    'folder': folder_name,
                    'type': img_type,
                    'success': False,
                    'error': str(e)
                })

        return results

    def generate_report(self, results):
        """生成处理报告"""
        report_path = self.output_root / "processing_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("图像处理报告\n")
            f.write("=" * 50 + "\n\n")

            successful = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]

            f.write(f"总处理数量: {len(results)}\n")
            f.write(f"成功: {len(successful)}\n")
            f.write(f"失败: {len(failed)}\n\n")

            if successful:
                f.write("成功处理的文件:\n")
                f.write("-" * 30 + "\n")
                for r in successful:
                    f.write(f"\n文件夹: {r['folder']}\n")
                    f.write(f"图像类型: {r['type']}\n")
                    f.write(f"对齐信息: {r['alignment_info']}\n")
                    f.write(f"裁剪信息: {r['crop_info']}\n")

            if failed:
                f.write("\n处理失败的文件:\n")
                f.write("-" * 30 + "\n")
                for r in failed:
                    f.write(f"\n文件夹: {r['folder']}\n")
                    f.write(f"图像类型: {r['type']}\n")
                    f.write(f"错误: {r['error']}\n")



# 初始化处理器
processor = ImagePairProcessor(
    perfusion_root="path/to/perfusion",
    target_root="path/to/target",
    output_root="path/to/output"
)

# 处理所有图像对
results = processor.process_all_pairs()

# 生成报告
processor.generate_report(results)