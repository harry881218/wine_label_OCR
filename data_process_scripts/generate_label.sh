# for file in /Users/harryliu/Desktop/Applied\ Computer\ Vision/project1/Wine\ Label\ Detection.v7-5th-run.coco/tmp/*; do
#     { printf "\n ${file##*/} " & tesseract "$file" stdout | tr '\n' ' ' ; }
# done

for file in /Users/harryliu/Desktop/Applied\ Computer\ Vision/project1/Wine\ Label\ Detection.v7-5th-run.coco/bbox_imgs/*; do
    file_name=$(printf "${file##*/} ")
    tess=$(tesseract "$file" stdout quiet | tr '\n' ' ') 
    printf '%s\n'  "$file_name $tess" >> bbox_labels.txt
done