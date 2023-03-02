# for file in /Users/harryliu/Desktop/Applied\ Computer\ Vision/project1/Wine\ Label\ Detection.v7-5th-run.coco/valid/*
# do
#     # if grep exit code is 1 (file not found in the text document), we delete the file
#     #[[ ! $(grep -x "${file##*/}" filenames.txt &> /dev/null) ]] && mv "$file" /Users/harryliu/Desktop/Applied\ Computer\ Vision/project1/Wine\ Label\ Detection.v7-5th-run.coco/not_used
#     echo "${file##*/}"
# done

#cat ../filenames.txt | xargs -I {} mv {} ../valid     #run this in the directory of images
cat ../filtered_bbox_valid_imgnames.txt | xargs -I {} mv {} ./valid