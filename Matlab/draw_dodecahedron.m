function [ fig, plt, tx ] = draw_dodecahedron( dode, newfig, cmd, txt, offset )
if newfig
    fig = figure();
    [X,Y,Z] = sphere();
    surf(X*sqrt(3),Y*sqrt(3),Z*sqrt(3),'FaceColor','yellow','EdgeColor','none')
%     alpha(.7)
    axis equal
    material dull;camlight;
else
    fig = gcf;
end
hold on
plt = plot3(dode(:,1),dode(:,2),dode(:,3), cmd);
if ~isempty(txt)
    tx = text(offset*dode(:,1),offset*dode(:,2),offset*dode(:,3),txt);
end
hold off
end

