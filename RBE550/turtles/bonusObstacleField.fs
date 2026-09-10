FeatureScript 3070;
import(path : "onshape/std/common.fs", version : "3070.0");

annotation { "Feature Type Name" : "Random Grid", "Feature Type Description" : "Generates a random grid of tetrominoes." }
export const randomGrid = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Grid Width (Cells)" }
        isInteger(definition.width, {(unitless): [4, 4, 128]} as IntegerBoundSpec);

        annotation { "Name" : "Grid Height (Cells)" }
        isInteger(definition.height, {(unitless): [4, 4, 128]} as IntegerBoundSpec);

        annotation { "Name" : "Cell Size" }
        isLength(definition.cellSize, LENGTH_BOUNDS);

        annotation { "Name" : "Coverage" }
        isReal(definition.coverage, {(unitless): [0, .5, 1]} as RealBoundSpec);

        annotation { "Name" : "Origin", "Filter" : (EntityType.VERTEX || BodyType.POINT) && SketchObject.YES, "MaxNumberOfPicks" : 1 }
        definition.gridOrigin is Query;
    }
    {
        var rng = makeRandomInt(id);

        var cellSize = definition.cellSize;
        const SQUARE_VERTS = [vector(0,0), vector(2,0), vector(2,2), vector(0,2), vector(0,0)];
        const L_VERTS = [vector(0,0), vector(0, 3), vector(1,3),
        vector(1,1), vector(2,1), vector(2,0), vector(0,0)];
        const BAR_VERTS = [vector(0,0), vector(4,0), vector(4,1), vector(0,1), vector(0,0)];
        const S_VERTS = [vector(0,0), vector(2,0), vector(2,1),
        vector(3,1), vector(3,2), vector(1,2),
        vector(1,1), vector(0,1), vector(0,0)];
        const T_VERTS = [vector(0,0), vector(3,0), vector(3,1),
        vector(2,1), vector(2,2), vector(1,2), vector(1,1),
        vector(0,1), vector(0,0)];

        var shapesList = [{"verts": SQUARE_VERTS, "dim": vector(2,2)},
        {"verts": L_VERTS, "dim": vector(2,3)}, // not bothering with mirrors
        {"verts": BAR_VERTS, "dim": vector(4,1)},
        {"verts": S_VERTS, "dim": vector(3,2)},
        {"verts": T_VERTS, "dim": vector(3,2)},];

        var rot2d = matrix([[0,1],[-1,0]]);

        var sketchPlane = evOwnerSketchPlane(context, {
                "entity" : definition.gridOrigin
        });
        var gridSketch = newSketchOnPlane(context, id + "gridSketch", {
                "sketchPlane" : sketchPlane
        });

        createGrid(context, id, gridSketch, sketchPlane, definition.gridOrigin, definition.width, definition.height, cellSize);

        var tetsToDraw = floor(definition.coverage * definition.width * definition.height / 4);
        println("tetsToDraw " ~ tetsToDraw);
        var coords = coordSystem(sketchPlane);
        var gridOriginRef = fromWorld(coords, evVertexPoint(context, {
                "vertex" : definition.gridOrigin
        }));

    gridOriginRef = vector(gridOriginRef[0]*(meter/cellSize)/meter, gridOriginRef[1]*(meter/cellSize)/meter); // drop z because it's 0 in plane, convert to grid cells
        for(var i = 0; i < tetsToDraw; i += 1)
        {
            var shape = shapesList[rng(0, size(shapesList))];
            var verts = shape["verts"];
            var dim = shape["dim"];
            var rotations = rng(0, 4);

            for(var r = 0; r < rotations; r += 1)
            {
                verts = mapArray(verts, x => rot2d*x);
                dim = rot2d*dim;
            }

            var xCell = rng(max(0, -dim[0]), min(definition.width + 1, definition.width - dim[0] + 1));
            var yCell = rng(max(0, -dim[1]), min(definition.height + 1, definition.height - dim[1] + 1));

            //println("(" ~ xCell ~ "," ~ yCell ~ ")");
            //println("dim " ~ dim);
            //println("rotations " ~ rotations);
            verts = mapArray(verts, x => (x + gridOriginRef + vector(xCell, yCell)) * cellSize);
            //println("verts " ~ verts);
            var tetSketch = newSketchOnPlane(context, id + ("tetrominoSketch" ~ i), {
                "sketchPlane" : sketchPlane
            });
            skPolyline(tetSketch, "polyline" ~ i, { "points": verts });

            skSolve(tetSketch);
            opExtrude(context, id + ("extrude" ~ i), // this goes faster with one sketch and one extrude, but then the tetrominoes lose their shapes
            {
                "entities": qCreatedBy(id + ("tetrominoSketch" ~ i), EntityType.FACE),
                "direction": sketchPlane.normal,
                "endBound": BoundingType.BLIND,
                "endDepth": cellSize
            });
        }


    });

export function createGrid(context is Context, id is Id, sketch is Sketch, sketchPlane is Plane, gridOrigin is Query, width is number, height is number, cellSize is ValueWithUnits)
{
    var coords = coordSystem(sketchPlane);

    var gridOriginRef = fromWorld(coords, evVertexPoint(context, {
            "vertex" : gridOrigin
    }));

    gridOriginRef = vector(gridOriginRef[0]*(meter/cellSize)/meter, gridOriginRef[1]*(meter/cellSize)/meter); // drop z because it's 0 in plane, convert to grid cells

    var columnLineEnds = [(vector(0,0) + gridOriginRef) * cellSize, (vector(0, height) + gridOriginRef) * cellSize];
    print(columnLineEnds);
    var rowLineEnds = [(vector(0,0) + gridOriginRef) * cellSize, (vector(width, 0) + gridOriginRef) * cellSize];
    print(rowLineEnds);

    for(var i = 0; i <= width; i += 1)
    {
        skLineSegment(sketch, "columnLine" ~ i, {
                "start" : columnLineEnds[0] + vector(i, 0) * cellSize,
                "end" : columnLineEnds[1] + vector(i, 0) * cellSize,
                "construction": true
        });
    }

    for(var i = 0; i <= height; i += 1)
    {
        skLineSegment(sketch, "rowLine" ~ i, {
                "start" : rowLineEnds[0] + vector(0, i) * cellSize,
                "end" : rowLineEnds[1] + vector(0, i) * cellSize,
                "construction": true
        });
    }

    skSolve(sketch);
}

// hacky RNG courtesy of OnShape blog https://www.onshape.com/en/resource-center/tech-tips/tech-tip-pseudo-random-number-generation-in-featurescript
const chrMap = {
'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6,
'H' : 7, 'I' : 8, 'J' : 9, 'K' : 10, 'L' : 11, 'M' : 12, 'N' : 13,
'O' : 14, 'P' : 15, 'Q' : 16, 'R' : 17, 'S' : 18, 'T' : 19, 'U' : 20,
'V' : 21, 'W' : 22, 'X' : 23, 'Y' : 24, 'Z' : 25,
'a' : 26, 'b' : 27, 'c' : 28, 'd' : 29, 'e' : 30, 'f' : 31, 'g' : 32,
'h' : 33, 'i' : 34, 'j' : 35, 'k' : 36, 'l' : 37, 'm' : 38, 'n' : 39,
'o' : 40, 'p' : 41, 'q' : 42, 'r' : 43, 's' : 44, 't' : 45, 'u' : 46,
'v' : 47, 'w' : 48, 'x' : 49, 'y' : 50, 'z' : 51,
'_' : 99, '-' : 98
};

function makeRandomInt(seed is Id) returns function
{
    const a = 214013;
    const c = 2531011;
    const m = 2 ^ 31 - 1;

    var chars = splitIntoCharacters(seed[0]);
    var seedNum = 0;
    for(var char in chars)
    {
        if(chrMap[char] != undefined)
        {
            seedNum = (seedNum + chrMap[char]) % 10000;
        }
    }

    var state = new box(seedNum);

    return function(min, max)
    {
        state[] = (a * state[] + c) % m;
        return state[] % (max - min) + min;
    };
}
