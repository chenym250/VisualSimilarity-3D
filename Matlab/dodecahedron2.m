close all;

vertices =   [
    [ 1.        ,  1.        ,  1.        ];
    [ 1.        ,  1.        , -1.        ];
    [ 1.        , -1.        ,  1.        ];
    [ 1.        , -1.        , -1.        ];
    [ 0.        ,  0.61803399,  1.61803399];
    [ 0.        ,  0.61803399, -1.61803399];
    [ 0.61803399,  1.61803399,  0.        ];
    [ 0.61803399, -1.61803399,  0.        ];
    [ 1.61803399,  0.        ,  0.61803399];
    [-1.61803399,  0.        ,  0.61803399];
    [-1.        , -1.        , -1.        ];
    [-1.        , -1.        ,  1.        ];
    [-1.        ,  1.        , -1.        ];
    [-1.        ,  1.        ,  1.        ];
    [ 0.        , -0.61803399, -1.61803399];
    [ 0.        , -0.61803399,  1.61803399];
    [-0.61803399, -1.61803399,  0.        ];
    [-0.61803399,  1.61803399,  0.        ];
    [-1.61803399,  0.        , -0.61803399];
    [ 1.61803399,  0.        , -0.61803399];
    ];
label = {'1','2','3','4','5','6','7','8','9','10',...
    '11','12','13','14','15','16','17','18','19','20'};
face1 = [10,12,16,5,14];
face2 = [17,8,3,16,12];
face3 = [11,17,12,10,19];
face4 = [15,4,8,17,11];
face5 = [6,15,11,19,13];
face6 = [13,19,10,14,18];
face7 = [15,6,2,20,4];
face8 = [6,13,18,7,2];
face9 = [2,7,1,9,20];
face10 = [18,14,5,1,7];
face11 = [9,1,5,16,3];
face12 = [4,20,9,3,8];
faces = [face1;face2;face3;face4;face5;face6;face7;face8;face9;face10;face11;face12];
figure()
patch('faces',faces,'vertices',vertices,'FaceColor','yellow')
material dull;camlight;view(45, 45);axis equal
draw_dodecahedron(vertices,false,'b.',label,1.1);

%% find all edges
adj_matrix = zeros(20);
for m = 1:12
    face = faces(m,:);
    adj_matrix(face(1),face(2)) = 1;
    adj_matrix(face(2),face(3)) = 1;
    adj_matrix(face(3),face(4)) = 1;
    adj_matrix(face(4),face(5)) = 1;
    adj_matrix(face(5),face(1)) = 1;
end
[row,col] = find(triu(adj_matrix));
% triu is the upper triangular matrix, so we only count each edge once
edges = [row,col];

%% draw the centers
centers = zeros(12,3);
for face_id = 1:12
    v1 = vertices(faces(face_id,1),:);
    v2 = vertices(faces(face_id,2),:);
    v3 = vertices(faces(face_id,3),:);
    v4 = vertices(faces(face_id,4),:);
    v5 = vertices(faces(face_id,5),:);
    v_mean = 0.2*(v1+v2+v3+v4+v5);
    centers(face_id,:) = v_mean;
end
% draw_dodecahedron(centers,false,'cx',label(1:12),1.1); % with face label
draw_dodecahedron(centers,false,'cx',{},1.1); % without face label
permutation = zeros(20,59);
permutation_index = 1;

%% 6x4 rotations along about the centers of face pairs
fltpnt = 100000;
for face_id = 1:6
    hold on
    center1 = centers(face_id,:);
    center2 = centers(face_id+6,:);
    ax = center2-center1; % rotate along this line which passes (0,0)
    h1 = plot3([-2*ax(1),2*ax(1)],[-2*ax(2),2*ax(2)],[-2*ax(3),2*ax(3)]); 
    % visualize this line
    [azi,ele,~] = cart2sph(ax(1),ax(2),ax(3));
    for rotate_id = 1:4
        new_verts = rot(vertices,azi,ele,2*rotate_id*pi/5); % rotate by 2pi/5
        [fig, plt,tx] = draw_dodecahedron(new_verts,false,'ro',label,1.3);
        [~,ia,ib] = intersect(round(vertices*fltpnt)/fltpnt,...
            round(new_verts*fltpnt)/fltpnt,'rows','stable');
        % remove the float point inconsistency so we can use intersect()
        % 'stable' meaning ia has the same order as dode (1:20)
        if length(ia) ~= 20
            raise('the polygon is not matched after rotation')
        end
        permutation(:,permutation_index) = ib;
        saveas(fig,['img\permutation_',num2str(permutation_index),'.png'])
        delete(plt)
        delete(tx)
        permutation_index = permutation_index + 1;

    end
    delete(h1)
end

%% 15x1 rotation along edge pairs
edge_direction = zeros(30,3);
for m = 1:30
    v1 = edges(m,1);
    v2 = edges(m,2);
    edge_direction(m,:) = vertices(v1,:) - vertices(v2,:);
end
edge_length = sqrt(5)-1;
% regular dodecahedron has uniform edge length of sqrt(5) - 1
dot_product = edge_direction*edge_direction'; % 30x30
is_parallel = abs(dot_product) > (edge_length^2 - 0.1) & ...
    abs(dot_product) < (edge_length^2 + 0.1);
[row,col] = find(triu(is_parallel,1)); % triu(...,1) to avoid diagonal
for m = 1:15
    ax = edge_direction(row(m),:); % can use either row or col here
    hold on
    h1 = plot3([-2*ax(1),2*ax(1)],[-2*ax(2),2*ax(2)],[-2*ax(3),2*ax(3)]);
    [azi,ele,~] = cart2sph(ax(1),ax(2),ax(3));
    new_verts = rot(vertices,azi,ele,pi); % rotate by pi (per pair of edges)
    [fig, plt,tx] = draw_dodecahedron(new_verts,false,'ro',label,1.3);
    [~,ia,ib] = intersect(round(vertices*fltpnt)/fltpnt,...
        round(new_verts*fltpnt)/fltpnt,'rows','stable');
    % remove the float point inconsistency so we can use intersect()
    % 'stable' meaning ia has the same order as dode (1:20)
    if length(ia) ~= 20
        raise('the polygon is not matched after rotation')
    end
    permutation(:,permutation_index) = ib;
    saveas(fig,['img\permutation_',num2str(permutation_index),'.png'])
    delete(plt)
    delete(tx)
    delete(h1)
    permutation_index = permutation_index + 1;
end

%% find vertex pairs
% meaning vertices that are parallel about the origin
% skipped because vertices is constructed in a way that 
% vertices(n,:) is parallel to ertices(n+10,:)
assert(isequal(vertices,[ vertices(1:10,:); -vertices(1:10,:)]))
%% 10x2 rotation along vertex pairs
for m = 1:10
    ax = vertices(m,:)-vertices(m+10,:);
    hold on
    h1 = plot3([-2*ax(1),2*ax(1)],[-2*ax(2),2*ax(2)],[-2*ax(3),2*ax(3)]);
    [azi,ele,~] = cart2sph(ax(1),ax(2),ax(3));
    for n = [-1,1]
        new_verts = rot(vertices,azi,ele,n*2*pi/3);
        % rotate by +/- 2pi/3
        [fig, plt,tx] = draw_dodecahedron(new_verts,false,'ro',label,1.3);
        [~,ia,ib] = intersect(round(vertices*fltpnt)/fltpnt,...
            round(new_verts*fltpnt)/fltpnt,'rows','stable');
        % remove the float point inconsistency so we can use intersect()
        % 'stable' meaning ia has the same order as dode (1:20)
        if length(ia) ~= 20
            raise('the polygon is not matched after rotation')
        end
        permutation(:,permutation_index) = ib;
        saveas(fig,['img\permutation_',num2str(permutation_index),'.png'])
        delete(plt)
        delete(tx)
        permutation_index = permutation_index + 1;
    end
    delete(h1)
end

%% finally, halve the vertices and permutation matrix
vertices_half = vertices(1:10,:); 
% the rest of the vertices are simply the negation of vertices_half
permutation_half = permutation(1:10,:);
permutation_half(permutation_half > 10) = ...
    permutation_half(permutation_half > 10) - 10;

%% save (part of) the workspace
[azi,ele] = cart2sph(vertices(:,1),vertices(:,2),vertices(:,3));
vertices_sph = [azi,ele];
% ele is the angle away from the x-y plane, from (-90,90)
[azi,ele] = cart2sph(vertices_half(:,1),vertices_half(:,2),vertices_half(:,3));
vertices_sph_half = [azi,ele];
save('dodecahedron_data.mat','vertices','faces','edges','permutation','vertices_sph')
save('dodecahedron_halfdata.mat','vertices_half','permutation_half','vertices_sph_half')