"""生成 OBS 测试场景集合：一个媒体源（视频循环播放，或静态图片）+ 自动马赛克滤镜。

用法：python3 tools/make-test-collection.py <输出.json> <视频或图片路径> [delay_ms] [--filter-off] [键=值 ...]
键=值 会写进滤镜设置，例如 nude_model=1 face_enabled=false nude_threshold=0.15
"""
import json
import sys
import uuid


def main():
    out, video = sys.argv[1], sys.argv[2]
    rest = sys.argv[3:]
    filter_on = "--filter-off" not in rest
    nums = [a for a in rest if not a.startswith("--") and "=" not in a]
    delay = int(nums[0]) if nums else 150
    settings = {"delay_ms": delay}
    for kv in (a for a in rest if "=" in a):
        k, v = kv.split("=", 1)
        if v in ("true", "false"):
            settings[k] = v == "true"
        else:
            for conv in (int, float, str):
                try:
                    settings[k] = conv(v)
                    break
                except ValueError:
                    pass
    is_image = video.lower().endswith((".jpg", ".jpeg", ".png"))
    source = ({"id": "image_source", "versioned_id": "image_source", "settings": {"file": video}} if is_image else
              {"id": "ffmpeg_source", "versioned_id": "ffmpeg_source",
               "settings": {"is_local_file": True, "local_file": video, "looping": True,
                            "restart_on_activate": True}})

    src, scene, flt = (str(uuid.uuid4()) for _ in range(3))
    d = {
        "name": "obs-ai-test",
        "current_scene": "人脸马赛克测试",
        "current_program_scene": "人脸马赛克测试",
        "scene_order": [{"name": "人脸马赛克测试"}],
        "sources": [
            {"id": "scene", "versioned_id": "scene", "name": "人脸马赛克测试", "uuid": scene,
             "settings": {"id_counter": 1, "custom_size": False, "items": [
                 {"name": "测试视频", "source_uuid": src, "visible": True, "locked": False, "id": 1,
                  "pos": {"x": 0.0, "y": 0.0}, "rot": 0.0, "scale": {"x": 1.0, "y": 1.0}, "align": 5,
                  "bounds_type": 2, "bounds_align": 0, "bounds": {"x": 1920.0, "y": 1080.0},
                  "crop_left": 0, "crop_top": 0, "crop_right": 0, "crop_bottom": 0}]}},
            dict(source, name="测试视频", uuid=src,
                 filters=[{"id": "face_mosaic_filter", "versioned_id": "face_mosaic_filter",
                           "name": "自动马赛克", "uuid": flt, "enabled": filter_on,
                           "settings": settings}]),
        ],
    }
    with open(out, "w") as fp:
        json.dump(d, fp, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
