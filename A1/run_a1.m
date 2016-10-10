% TODO: Buffet_Lettuce was turned into a greyscale image... ummm...
% Also, just deal with the fact that there may be grey images - for these,
% ignore when using colour
% TODO: modifying k -- can just do it with mdl.NumNeighbors = <newval>
% instead of making model again!

start_path = '/Users/karinng/Documents/1UniHW/cs9840/Assignments/A1/TenCategories';
total_img_num = 0;
img_size = 50;
num_bins = 50;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up new resized images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make new directory for resized images (if needed)
[top_path, img_fol, ext] = fileparts(start_path);
resized_dir = fullfile(top_path, 'resized_imgs');
if ~exist(resized_dir, 'dir')
    mkdir(resized_dir);
end

% Loop over the subdirectories (ignoring current and previous)
cat_subdirs = dir(start_path);
cat_subdirs = cat_subdirs(~ismember({cat_subdirs.name}, {'.', '..'}));
for category_idx = 1:numel(cat_subdirs)
    category = cat_subdirs(category_idx).name;
    
    % Make new subdirectory folder if needed
    new_cat_fol = fullfile(resized_dir, category);
    if ~exist(new_cat_fol, 'dir')
        mkdir(fullfile(resized_dir, category));
    end
    
    % Resizing and saving the newly sized images
    imgs = dir(fullfile(start_path, category, '*.jpg'));
    for img_idx = 1:numel(imgs)
        img_name = imgs(img_idx).name;
        
        % Save the resized images (if don't already exist)
        if ~exist(fullfile(resized_dir, category, img_name), 'file')
            img = imread(fullfile(start_path, category, img_name));
            resized_img = imresize(img, [img_size img_size]);
            imwrite(resized_img, fullfile(resized_dir, category, img_name)); 
        end
    end
    
    total_img_num = total_img_num + numel(imgs);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p1a(total_img_num, img_size, resized_dir, cat_subdirs);
% p1b(total_img_num, img_size, resized_dir, cat_subdirs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p2(total_img_num, num_bins, start_path, cat_subdirs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p3(total_img_num, start_path, cat_subdirs);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p4(total_img_num, start_path, cat_subdirs);



