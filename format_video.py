from evaluate import format_video
import json
#format video from orgin data
if __name__ == "__main__":
    path = "/data/longshaohua/IVCR_2/output/test_for_final_ivcr_VR/2w_loss_superparam_001_use_3651_didemo/IVCR_test_f96_result.json"
    with open(path, 'r') as file:
        data = json.load(file)
    result = format_video(data)
    with open("/data/longshaohua/IVCR_2/output/test_for_final_ivcr_VR/2w_loss_superparam_001_use_3651_didemo/fmt_IVCR_test_f96_result.json",'w') as file:
        json.dump(result, file, indent=4)




