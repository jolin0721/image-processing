% Load the encoded data
load('jpegcodes.mat', 'DC_code', 'AC_code', 'imageHeight', 'imageWidth');
load('JpegCoeff.mat');

% Parameters
blockSize = 8;
numBlocks = 15*21;

% Initialize the decoded DC and AC coefficients
DC_coeffs_decoded = zeros(1, numBlocks);
AC_coeffs_decoded = zeros(numBlocks, 63);

% Huffman DC decoding (simplified based on given huffman table)
% step 1: reading bits by bits from the DC_code, 
% checking starting from the column 2, until there is a row that is
% completely same with the bits we read, then we check which row it is, 
% then we know the category(row-1)
% step 2: checking the column 1 of ACTAB matrix in JpegCoeff.mat
% then we will get the length of magnitude, we double check whether it is 
% equal to the bits we have read
% step 3: by knowing the category, we now extract the category number of 
% bits after the bits we read as magnitude
% step 4: bin2dec(magnitude), stored it and starting from the second one
% y(n) = x(n-1) - x(n)

% DC Huffman decoding
index = 1;  % Initialize index for reading bits

for i=1:315
%     % Decode the first DC coefficient
%     category = 0;
%     found_match = false;  % Flag to determine if a match is found
% 
%     while ~found_match
%         % Read bits progressively for the current category
% %         if index + category > length(DC_code)
% %             error('Index exceeds the number of elements in DC_code.');
% %         end
%         if category == 0
%             current_bits = DC_code(index:index);
%         else 
%             current_bits = DC_code(index:index+category-1);
%         end
%         
%         % Convert current_bits to a string
%         current_bits_str = num2str(current_bits');  % Convert numeric array to string
% 
%         % Manually compare each row in DCTAB to find a match
%         for row = 1:12
%             huffman_length = DCTAB(row, 1);  % Length of the Huffman code in this row
%             huffman_code = DCTAB(row, 2:huffman_length+1);  % The actual Huffman code
%             
%             % Convert Huffman code to a string for comparison
%             huffman_code_str = num2str(huffman_code');
% 
%             if isequal(huffman_code_str, current_bits_str)  % Compare with current bits
%                 category = row - 1;  % Found the category
%                 found_match = true;
%                 break;
%             end
%         end
%         
%         % Increase the bit count if no match is found
%         if ~found_match
%             category = category + 1;
%         end
%     end
    
    for category = 0:11
        huffman_length = DCTAB(category+1,1);
        huffman_code = DCTAB(category+1, 2:huffman_length+1);

        % Convert Huffman code to a string for comparison
        huffman_code_str = num2str(huffman_code');
        current_bits = DC_code(index:index+huffman_length-1);
        current_bits_str = num2str(current_bits'); 

        if isequal(huffman_code_str, current_bits_str)
            break;
        end
    end

    % Move the index to the next bit after the Huffman code
%     index = index + length(num2str(DCTAB(row, 2:huffman_length+1)'));
    index = index + huffman_length;
    % Extract magnitude bits
%     if index + category - 1 > length(DC_code)
%         error('Index exceeds the number of elements in DC_code while extracting magnitude bits.');
%     end
    if category == 0
        magnitude_bits = DC_code(index:index);
    else
        magnitude_bits = DC_code(index:index+category-1);
    end

    % Initialize magnitude value
    magnitude_value = 0;

    % Check if magnitude_bits is empty
    if isempty(magnitude_bits)
        error('Magnitude bits are empty.');
    end

    % Decode the magnitude using the provided method
    if magnitude_bits(1) == 0
        % If the first bit is '0', it's a negative value; apply two's complement manually
        for counting = 1:category
            magnitude_value = magnitude_value + (1 - magnitude_bits(counting)) * 2^(category-counting);
        end
        magnitude_value = -magnitude_value;
    else
        % Otherwise, it's a positive value
        for counting2 = 1:category
            magnitude_value = magnitude_value + magnitude_bits(counting2) * 2^(category-counting2);
        end
        if category == 0
            magnitude_value = magnitude_bits(1)*1;
        end
    end
    
    if category == 0
        index = index + 1;
    else
        index = index + category;
    end
%     % Display the results for debugging
%     disp('Index after Huffman code:');
%     disp(index);
% % 
%     disp('Category:');
%     disp(category);
% % 
%     disp('Magnitude bits:');
%     disp(magnitude_bits);
% % 
%     disp('Decoded magnitude value:');
%     disp(magnitude_value);

    % Assign the first decoded DC coefficient
    DC_coeffs_decoded(i) = magnitude_value;
end

% Initialize the first DC coefficient
DC_coeffs_decoded_final(1) = DC_coeffs_decoded(1);
% disp(DC_coeffs_decoded_final(1));
% Perform differential decoding for the remaining DC coefficients
for i = 2:315
    % Add the difference to the previous decoded coefficient
    DC_coeffs_decoded_final(i) = DC_coeffs_decoded_final(i-1) - DC_coeffs_decoded(i);
end

% Display the final DC coefficients
% disp(DC_coeffs_decoded_final);


% Huffman AC decoding
index = 1;  % Initialize index for reading bits

for blockIdx = 1:315
    k = 1;  % Initialize position in the block for non-DC coefficients (from 2 to 64)
    
%     disp("heyheyheyeheeeeeeeeeeeeeeeeeeeeeee");
    while k <= 63
        extreme_case = false;
        found_match = false;  % Flag to determine if a match is found
        for row = 1:160
            huffman_length = ACTAB(row, 3);  % Length of the Huffman code in this row
            huffman_code = ACTAB(row, 4:3+huffman_length);  % The actual Huffman code

            % Convert Huffman code to a string for comparison
            huffman_code_str = num2str(huffman_code');
            if huffman_length > length(AC_code)-index+1
                continue
            end

            current_bits = AC_code(index:index+huffman_length-1);  % Extract current bits
            current_bits_str = num2str(current_bits');  % Convert to string for comparison

            % EOB case: fill the rest of the 63 positions with 0
            if strncmp(current_bits_str, '1010', 4)
                AC_coeffs_decoded(blockIdx, k:63) = 0;
%                 disp('EOB found, filling the rest of the block with zeros.');
                index = index + 4;
                found_match = true;
                extreme_case = true;
%                 for runrunrun = k:63
%                     disp(0);
%                 end
                k = 64; % Exit the loop
                break;
            end

            % ZRL case: fill the next 16 coefficients with 0
            if strncmp(current_bits_str, '11111111001', 11)
                AC_coeffs_decoded(blockIdx, k:k+15) = 0;
%                 disp('ZRL found, skipping 16 coefficients.');
                index = index + 11;
                k = k + 16;
                found_match = true;
                extreme_case = true;
%                 if k > 63
%                     disp('k exceeds 63 after ZRL, check for errors.');
%                     break;
%                 end
%                 for runrunrun = 1:16
%                     disp(0);
%                 end
                continue;
            end

            if isequal(huffman_code_str, current_bits_str)  % Compare with current bits
                runLength = ACTAB(row, 1);  % Extract run length (first column)
                sizeAC = ACTAB(row, 2);     % Extract size (second column)
                found_match = true;
                break;
            end
        end
        
        if ~found_match
            disp('No Huffman code match found, check for errors.');
            break;
        end

        % Step 2: Move the index to the next position after the Huffman code
%         if extreme_case == false
%             index = index + huffman_length;
%         end

        % Check for magnitude length 0 (shouldn't happen normally)
        if sizeAC == 0
            disp('Warning: Magnitude length is zero, skipping this entry.');
            continue;
        end
    if extreme_case == false
        index = index + huffman_length;
        % Step 4: Extract the magnitude bits
        magnitude_bits = AC_code(index:index+sizeAC-1);
        index = index + sizeAC;  % Move index to the next position after magnitude bits

        % Step 5: Decode the magnitude value
        magnitude_value = 0;
        if magnitude_bits(1) == 0
            % Negative value: apply two's complement manually
            for i = 1:sizeAC
                magnitude_value = magnitude_value + (1 - magnitude_bits(i)) * 2^(sizeAC-i);
            end
            magnitude_value = -magnitude_value;
        else
            % Positive value
            for i = 1:sizeAC
                magnitude_value = magnitude_value + magnitude_bits(i) * 2^(sizeAC-i);
            end
        end

        % Step 6: Place the decoded AC coefficient in the correct position
        if runLength > 0
            AC_coeffs_decoded(blockIdx, k:k+runLength-1) = 0;
%             disp(['Run length of ', num2str(runLength), ' zeros inserted at position ', num2str(k)]);
%             for runrunrun = 1:runLength
%                 disp(0);
%             end
        end
        
        k = k + runLength;  % Skip the runLength number of zeros

%         if k > 63
%             disp('k exceeds 63 after placing zeros, check for errors.');
%             break;
%         end

        AC_coeffs_decoded(blockIdx, k) = magnitude_value;  % Place the coefficient
%         disp(['AC coefficient placed at position ', num2str(k), ': ', num2str(magnitude_value)]);
        k = k + 1;  % Move to the next coefficient position
    end
%         if k > 63
%             disp('k exceeds 63 after placing a coefficient, check for errors.');
%             break;
%         end
%         disp(magnitude_value);
    end
    
%     disp(blockIdx);
%     disp(index);
end

% disp(AC_coeffs_decoded);


% Reconstruct the image blocks
reconstructedBlocks = zeros(numBlocks, 8, 8);
for i = 1:numBlocks
    % Inverse quantization
%     reconstructedMatrix = inverseZigzag(AC_coeffs_decoded);
%     quantizedBlock = reconstructedMatrix;
     quantizedBlock = [DC_coeffs_decoded_final(i) AC_coeffs_decoded(i, :)];
%      disp(quantizedBlock);
     shapingBlock = inverseZigzag(quantizedBlock);
%      disp(shapingBlock);
%     inverseQuantBlock = reshape(quantizedBlock, [8, 8]) .* quantMatrix;
    inverseQuantBlock = shapingBlock .* quantMatrix;
    % Apply inverse DCT
    reconstructedBlock = idct2(inverseQuantBlock);
%      disp(reconstructedBlock);
    % Add 128 to shift back
    reconstructedBlocks(i,:,:) = reconstructedBlock;
%     reconstructedBlocks(i, :, :) = reconstructedBlock + 128;
end
% for iii = 1:315
%     disp(reconstructedBlocks(iii,:,:));
% end
% Stitch the blocks back together into the image
reconstructedImage = zeros(120, 168);
blockIndex = 1;
for i = 1:8:120
    for j = 1:8:168
        reconstructedImage(i:i+7,j:j+7) = squeeze(reconstructedBlocks(blockIndex, :, :))+128;
        blockIndex = blockIndex + 1;
    end
end

% Convert the image to uint8 format
reconstructedImage = uint8(reconstructedImage);
% figure;
% imshow(reconstructedImage);

% Calculate PSNR
originalImage = hall_gray;
mse = mean((double(originalImage(:)) - double(reconstructedImage(:))).^2);
psnrValue = 10 * log10(255^2 / mse);

% Display results
figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
imshow(reconstructedImage);
title(['Reconstructed Image, PSNR: ', num2str(psnrValue), ' dB']);

disp(['PSNR between original and reconstructed image: ', num2str(psnrValue), ' dB']);
disp(['the ratio is: ',num2str(120*168*8/(length(AC_code)+length(DC_code)))]);

function matrix = inverseZigzag(zigzagOrder)
    rows = 8; % 8x8 matrix
    cols = 8;
    matrix = zeros(rows, cols);
    index = 1;

    for sum = 1:(rows + cols - 1)
        if mod(sum, 2) == 0
            r = min(sum, rows) - 1;
            c = sum - r - 1;
            while r >= 0 && c < cols
                matrix(r + 1, c + 1) = zigzagOrder(index);
                index = index + 1;
                r = r - 1;
                c = c + 1;
            end
        else
            c = min(sum, cols) - 1;
            r = sum - c - 1;
            while c >= 0 && r < rows
                matrix(r + 1, c + 1) = zigzagOrder(index);
                index = index + 1;
                c = c - 1;
                r = r + 1;
            end
        end
    end
end
