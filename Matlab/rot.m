function [ output ] = rot( input, azi, ele, alpha )
output = (rotz(-azi)*roty(ele)*rotx(alpha)*roty(-ele)*rotz(azi)*input')';
end


function [ Rx ] = rotx( rad )
Rx = [[1,0,0];[0,cos(rad),sin(rad)];[0,-sin(rad),cos(rad)]];

end
function [ Ry ] = roty( rad )
Ry = [[cos(rad),0,-sin(rad)];[0,1,0];[sin(rad),0,cos(rad)]];
end
function [ Rz ] = rotz( rad )
Rz = [[cos(rad),sin(rad),0];[-sin(rad),cos(rad),0];[0,0,1]];
end
