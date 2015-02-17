% Estimating rotation and translation using RANSAC
function [R,T,status] = prosecutes_analysis(data)

% Getting prosecutes analysis on it done
pt1 = data(:,1:3); pt2 =data(:,4:6);
tf_pt1 = pt1-repmat(mean(pt1),size(pt2,1),1);
tf_pt2 = pt2- repmat(mean(pt2),size(pt2,1),1);
H = tf_pt1'*tf_pt2;
[U,S,V] = svd(H);
R = V*U';
if (abs(det(R) + 1.0000) < 0.0002) && (S(3,3)==0)
    % Modifying v to check the case when this fails
    V(:,3) = -V(:,3);
    status = 1;
elseif (abs(det(R) + 1.0000) < 0.0002) && (S(3,3)>0)
    status = 0;
    disp('Rotation and Translation estimation failed');
else
    status  = 1;
end
R = V*U';
% Translation
T = mean(pt2)' - R*mean(pt1)';

end

