

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% make changes:
% add glucose, insulin, bolus and basal
% also need to test with the remaining patients 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
a = [559,563,570,575,588,591];
b = [0.15, 0.034, 0.011, 0.45, 0.086, 0.081];
for i = 1:6
    write_test_data(a(i));
    write_train_data(a(i));
    A_one_shot_7(a(i), b(i));
end
%write_test_data(a(1));
%write_train_data(a(1));
%A_one_shot_7(a(1), b(1));

