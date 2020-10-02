clear; clc;
%% Exact Solution
% PDE iii) u(x) = -f*x^2/(2*E*A) + (h+f*L)*x/(E*A)
%% Initilization
L = 0.1;
A = 1e-4;
E = 1e11;
g1 = 0;
g2 = 0.001;
h = 1e6;
q = 2;
f = 1e7;
%% Meshing
% Prompt asking user whether to create a mesh using 3 or 100 elements
% Nelem is total number of elements
% Le is lenth of each element
elemPrompt = 'Choose either 3 or 100 for the number of elements in the mesh: ';
elemInput = input(elemPrompt);
if elemInput == 3
    Nelem = 3;
    Le = L/Nelem;
else
    Nelem = 100;
    Le = L/Nelem;
end

% Prompt asking user whether to use Linear or Quad Langrange Shape Fxns
% N_en is number of nodes per element
% coord is Nodal coordinate array
% elements is Element Connectivity array
nenPrompt = 'Type 1 for Linear Elements or 2 for Quadratic Elements: ';
nenInput = input(nenPrompt);
if nenInput == 1
    N_en = 2;
    Nnodes = Nelem + 1;
    coord = (0:Le:L);
    elements = [];
    for i = 1:Nelem
        elements = [elements;[i,i+1]];
    end
else
    N_en = 3;
    Nnodes = 2*Nelem + 1;
    coord = (0:L/(Nnodes-1):L);
    elements = [];
    for i = 1:Nelem
        j = 2*i - 1;
        elements = [elements;[j,j+1,j+2]];
    end
end

% The Global matrices are initialized with the last value of the global D
% matrix set to g2
Kglobal = zeros(Nnodes, Nnodes);
Fglobal = zeros(Nnodes,1);
dglobal = zeros(Nnodes,1);


%% Element Loop

nodeCoords = zeros(N_en,1); %Used to showcase the coords of each node for each element
for elem = 1:Nelem
    
    %gets the coordinates for the nodes on this element
    for i=1:N_en
        nodeCoords(i) = coord(elements(elem,i));
    end
    
    % Intialization of Local Matrices of each element
    Klocal = zeros(N_en,N_en);
    Flocal = zeros(N_en,1);
    Dlocal = zeros(N_en,1);
    
    [z,w]=quad1(q); %obtains the z points and weights based on quad points used
    
    for int = 1:q
        [Nz,Bz,Jac,x_z] = shape(N_en,z(q),nodeCoords); %obtains the Shape Fxn and Gradient Matrix plus Jacobian and x,z value.
        Klocal = Klocal + E*A/Le*Jac*1.0/x_z*1.0/x_z*w(q)*Bz;
        Flocal = Flocal + A*f*z(q)*Nz*Jac*w(q);
    end
    
    %Assmebly of Local stifness into Global
    for i=1:N_en
        for j=1:N_en
            row = elements(elem,i);
            col = elements(elem,j);
            Kglobal(row,col) = Kglobal(row,col) + Klocal(i,j);
        end
    end
    
    %Assembly of Local force into Global
    for i=1:N_en
        row = elements(elem,i);
        Fglobal(row) = Fglobal(row) + Flocal(i);
    end
end


%% Global to Dirichlet
Kcopy = Kglobal;
Kcopy(1,:) = [];
Kcopy(:,1) = [];
Kd = Kcopy;

Fcopy = Fglobal;
Fcopy(1) = [];
Fcopy(Nnodes-1) = Fcopy(Nnodes-1) + A*h;
Fd = Fcopy;

dcopy = dglobal;
dcopy(1) = [];
dd = dcopy;

% Solves for Dirichlet D matrix
dd = Kd\Fd;

%% Post Processing

for i = 2:Nnodes
    dglobal(i) = dd(i-1);
end

plot(coord,dglobal, '-x');

xlabel('Nodal Coordinates (m)');
ylabel('Displacement (m)');

if elemInput == 3
    if nenInput == 1
        title('PDE iii) with 3 Linear Elements');
    elseif nenInput == 2
        title('PDE iii) with 3 Quad Elements');
    end
elseif elemInput == 100
    if nenInput == 1
        title('PDE iii) with 100 Linear Elements');
    elseif nenInput == 2
        title('PDE iii) with 100 Quad Elements');
    end
end

%% Gauss Qaudrature Function

%Function which creates cell arrays of the z-points and weight values for
%based on number of quad points upto 3.
function [point,weight] = quad1(q)

point = zeros(q,1);
weight = zeros(q,1);

if q == 1
    point(1) = 0.0;
    weight(1) = 2.0;
    
elseif q == 2
    point(1) = -1/sqrt(3);
    point(2) = -point(1);
    weight(1) = 1.0;
    weight(2) = weight(1);
    
else
    point(1) = -sqrt(3/5);
    point(2) = 0;
    point(3) = -point(1);
    weight(1) = 5/9;
    weight(2) = 8/9;
    weight(3) = weight(1);
    
end
end

%% Shape Function (N) and Gradient (B) Generator
%%Function which generates the Shape Functions and Gradient Matrices based
%%on whether Linear or Quad shape fxns are used. Takes in a singe z-value
%%of a quad point and the nodal coordinates of the nodes of an element.
function [N,B,Jac,x_z] = shape(N_en,z, nodeCoords)

%Linear Shape Fxns
if N_en == 2
    
    N1 = (1-z)/2.0;
    N2 = (1+z)/2.0;
    N = [N1;N2];
    
    B1 = (-1.0/2.0);
    B2 = (1.0/2.0);
    B = [B1*B1,B1*B2;B1*B2,B2*B2];
    
    x_z = (B1*nodeCoords(1) + B2*nodeCoords(2));
    Jac = abs(x_z); %Jacobian
    
    %Quadratic Shape Functions
elseif N_en == 3
    N1 = (-1.0/2.0)*(z-z^2);
    N2 = (1.0-z)*(1.0+z);
    N3 = (1.0/2.0)*(z+z^2);
    N = [N1;N2;N3];
    
    B1 = (-1.0/2.0)*(1.0-2.0*z);
    B2 = -2.0*z;
    B3 = (1.0+2.0*z)*(1.0/2.0);
    
    B = [B1*B1, B1*B2, B1*B3;B2*B1, B2*B2, B2*B3; B3*B1, B3*B2, B3*B3];
    
    x_z = B1*nodeCoords(1) + B2*nodeCoords(2) + B3*nodeCoords(3);
    
    Jac = abs(x_z);
    
end
return;
end
