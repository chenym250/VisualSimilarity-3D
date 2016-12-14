load('dodecahedron_data.mat')
N = 10;
best_d = 0;
best_v = zeros(10*20,3);
for repeat = 1:100
    a1 = 2*pi*rand(N^3,1);
    a2 = 2*pi*rand(N^3,1);
    a3 = 2*pi*rand(N^3,1);
    angles = zeros(9,3);
    angles_ = zeros(9,3);
    v_full = [vertices; zeros(9*20,3)];
    for n = 1:9
        d_min = zeros(N^3,1);
        for m = 1:N^3
            v2 = rot(vertices, a1(m),a2(m),a3(m));
            distance = acos(v2*v_full(1:n*20,:)'/3);
            d_min(m) = min(distance(:));
        end
        m0 = find(d_min == max(d_min),1);
        angle = [a1(m0),a2(m0),a3(m0)];
        angles(n,:) = angle;
        v0 = rot(vertices, a1(m0),a2(m0),a3(m0));
        v_full(n*20+1:(n+1)*20,:) = v0;
    end

    index = find(d_min == max(d_min),1);
    d3 = triu(acos(v_full*v_full'/3),1);
    d4 = d3+tril(ones(length(v_full)))*1000;
    min_d = min(d4(:));
    if min_d > best_d
        best_d = min_d;
        best_v = v_full;
    end

end

%% plot

label = {'1','2','3','4','5','6','7','8','9','10',...
    '11','12','13','14','15','16','17','18','19','20'};
draw_dodecahedron(vertices,true,'b.',label,1.1);
for n = 1:10
    draw_dodecahedron(best_v((n-1)*20+1:n*20,:),false,'r.',{});
end
figure()
v_prj = [best_v(:,1).*sqrt(2./(sqrt(3)-best_v(:,3))),best_v(:,2).*sqrt(2./(sqrt(3)-best_v(:,3)))];
plot(v_prj(:,1),v_prj(:,2),'color',[1,1,0],'LineStyle','.')