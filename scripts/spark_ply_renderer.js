#!/usr/bin/env node
/* eslint-disable no-console */
/**
 * Spark.js-backed renderer for still frames and traversal videos.
 */

const fs = require('fs');
const path = require('path');
const minimist = require('minimist');
const { spawn } = require('child_process');
const { Canvas } = require('skia-canvas');

const sparkRoot = path.dirname(require.resolve('sparkjs/package.json'));
const sparkLibDir = path.join(sparkRoot, 'app', 'lib');

const prevLib = global.lib;
global.lib = function loadSparkLib(libName) {
    return require(path.join(sparkLibDir, libName));
};

const CanvasAPI = require(path.join(sparkLibDir, 'apis/canvas'));
const MathAPI = require(path.join(sparkLibDir, 'apis/math'));
const ColorsAPI = require(path.join(sparkLibDir, 'apis/colors'));
const StringAPI = require(path.join(sparkLibDir, 'apis/string'));

if (typeof prevLib === 'undefined') {
    delete global.lib;
} else {
    global.lib = prevLib;
}

const JSContext = require(path.join(sparkLibDir, 'js_context'));

const COLOR_FIELDS = [
    ['red', 'green', 'blue'],
    ['diffuse_red', 'diffuse_green', 'diffuse_blue'],
    ['r', 'g', 'b'],
];
const SH_COLOR_FIELDS = ['f_dc_0', 'f_dc_1', 'f_dc_2'];
const ROT_FIELDS = ['rot_0', 'rot_1', 'rot_2', 'rot_3'];
const SCALE_FIELDS = ['scale_0', 'scale_1', 'scale_2'];
const OPACITY_FIELD = 'opacity';

const BINARY_READERS = {
    char: { size: 1, read: (buf, offset) => buf.readInt8(offset) },
    uchar: { size: 1, read: (buf, offset) => buf.readUInt8(offset) },
    int8: { size: 1, read: (buf, offset) => buf.readInt8(offset) },
    uint8: { size: 1, read: (buf, offset) => buf.readUInt8(offset) },
    short: { size: 2, read: (buf, offset) => buf.readInt16LE(offset) },
    ushort: { size: 2, read: (buf, offset) => buf.readUInt16LE(offset) },
    int16: { size: 2, read: (buf, offset) => buf.readInt16LE(offset) },
    uint16: { size: 2, read: (buf, offset) => buf.readUInt16LE(offset) },
    int: { size: 4, read: (buf, offset) => buf.readInt32LE(offset) },
    uint: { size: 4, read: (buf, offset) => buf.readUInt32LE(offset) },
    int32: { size: 4, read: (buf, offset) => buf.readInt32LE(offset) },
    uint32: { size: 4, read: (buf, offset) => buf.readUInt32LE(offset) },
    float: { size: 4, read: (buf, offset) => buf.readFloatLE(offset) },
    float32: { size: 4, read: (buf, offset) => buf.readFloatLE(offset) },
    double: { size: 8, read: (buf, offset) => buf.readDoubleLE(offset) },
    float64: { size: 8, read: (buf, offset) => buf.readDoubleLE(offset) },
    double64: { size: 8, read: (buf, offset) => buf.readDoubleLE(offset) },
};

const DEFAULTS = {
    width: 1920,
    height: 1080,
    fov: 60,
    background: '#050505',
    maxPoints: 0,
    pointScale: 1600,
    minPointSize: 1.0,
    maxPointSize: 14.0,
    minAlpha: 0.08,
    maxAlpha: 0.98,
    margin: 32,
    fps: 12,
};

function coerceNumber(value, fallback) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function parseVector(values, fallback) {
    if (!values) return fallback.slice();
    if (Array.isArray(values)) {
        if (values.length !== 3) {
            throw new Error('Vector must have exactly 3 components.');
        }
        return values.map((v) => Number(v));
    }
    const parts = values.split(',').map((p) => Number(p.trim()));
    if (parts.length !== 3 || parts.some((v) => Number.isNaN(v))) {
        throw new Error(`Invalid vector "${values}" – expected "x,y,z".`);
    }
    return parts;
}

function subtract(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

function length(v) {
    return Math.sqrt(dot(v, v));
}

function normalize(v) {
    const len = length(v);
    if (len < 1e-8) {
        return [0, 0, 0];
    }
    return [v[0] / len, v[1] / len, v[2] / len];
}

function sigmoid(value) {
    return 1.0 / (1.0 + Math.exp(-value));
}

function quaternionToMatrix(qw, qx, qy, qz) {
    const n = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) || 1.0;
    const w = qw / n;
    const x = qx / n;
    const y = qy / n;
    const z = qz / n;
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const xz = x * z;
    const yz = y * z;
    const wx = w * x;
    const wy = w * y;
    const wz = w * z;
    return [
        1 - 2 * (yy + zz),
        2 * (xy - wz),
        2 * (xz + wy),
        2 * (xy + wz),
        1 - 2 * (xx + zz),
        2 * (yz - wx),
        2 * (xz - wy),
        2 * (yz + wx),
        1 - 2 * (xx + yy),
    ];
}

function findHeaderEnd(buffer) {
    const marker = Buffer.from('end_header');
    const idx = buffer.indexOf(marker);
    if (idx === -1) {
        throw new Error('Malformed PLY header – missing "end_header".');
    }
    let offset = idx + marker.length;
    while (offset < buffer.length && buffer[offset] !== 0x0a) {
        if (buffer[offset] === 0x0d) {
            offset += 1;
            continue;
        }
        offset += 1;
    }
    if (offset < buffer.length) {
        offset += 1;
    }
    return offset;
}

function parsePlyHeader(buffer) {
    const headerEnd = findHeaderEnd(buffer);
    const headerText = buffer.slice(0, headerEnd).toString('utf8');
    const lines = headerText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (!lines.length || lines[0] !== 'ply') {
        throw new Error('Not a valid PLY file – missing leading "ply" line.');
    }

    let format = null;
    let vertexCount = 0;
    let parsingVertex = false;
    const vertexProperties = [];

    for (let i = 1; i < lines.length; i += 1) {
        const line = lines[i];
        if (line === 'end_header') {
            break;
        }
        if (line.startsWith('comment') || !line) {
            continue;
        }
        const parts = line.split(/\s+/);
        const keyword = parts[0];
        if (keyword === 'format') {
            format = parts.slice(1).join(' ');
        } else if (keyword === 'element') {
            parsingVertex = parts[1] === 'vertex';
            if (parsingVertex) {
                vertexCount = Number(parts[2]);
            }
        } else if (keyword === 'property' && parsingVertex) {
            if (parts[1] === 'list') {
                throw new Error(`List properties are not supported (${line}).`);
            }
            vertexProperties.push({
                type: parts[1],
                name: parts[2],
            });
        }
    }

    if (!format) {
        throw new Error('PLY header missing "format" line.');
    }
    if (vertexCount <= 0) {
        throw new Error('PLY file does not describe any vertices.');
    }

    const required = ['x', 'y', 'z'];
    const propNames = vertexProperties.map((prop) => prop.name);
    required.forEach((name) => {
        if (!propNames.includes(name)) {
            throw new Error(`PLY vertex element missing "${name}" property.`);
        }
    });

    return {
        format: format.toLowerCase(),
        vertexCount,
        vertexProperties,
        headerLength: headerEnd,
    };
}

function determineColorMode(propNames) {
    for (const candidate of COLOR_FIELDS) {
        if (candidate.every((name) => propNames.includes(name))) {
            return { mode: 'rgb', fields: candidate };
        }
    }
    if (SH_COLOR_FIELDS.every((name) => propNames.includes(name))) {
        return { mode: 'sh', fields: SH_COLOR_FIELDS };
    }
    return { mode: 'none', fields: [] };
}

function buildColorFieldMap(colorMode) {
    if (colorMode.mode !== 'rgb') {
        return {};
    }
    return colorMode.fields.reduce((acc, name, idx) => {
        acc[name] = idx;
        return acc;
    }, {});
}

function buildShFieldMap(colorMode) {
    if (colorMode.mode !== 'sh') {
        return null;
    }
    return SH_COLOR_FIELDS.reduce((acc, name, idx) => {
        acc[name] = idx;
        return acc;
    }, {});
}

function normalizeScale(value) {
    if (!Number.isFinite(value)) {
        return 0.01;
    }
    if (value <= 0 || Math.abs(value) <= 3.0) {
        return Math.exp(value);
    }
    return Math.abs(value);
}

function normalizeOpacity(value) {
    if (!Number.isFinite(value)) {
        return 1.0;
    }
    if (value >= 0.0 && value <= 1.0) {
        return value;
    }
    return sigmoid(value);
}

function parsePly(buffer) {
    const header = parsePlyHeader(buffer);
    const names = header.vertexProperties.map((p) => p.name);
    const colorMode = determineColorMode(names);

    const fieldMap = {
        rotations: ROT_FIELDS.filter((name) => names.includes(name)),
        scales: SCALE_FIELDS.filter((name) => names.includes(name)),
        opacity: names.includes(OPACITY_FIELD) ? OPACITY_FIELD : null,
    };

    if (header.format.startsWith('ascii')) {
        const dataText = buffer.slice(header.headerLength).toString('utf8');
        return parseAsciiVertices(dataText, header, colorMode, fieldMap);
    }
    if (header.format.startsWith('binary_little_endian')) {
        const dataBuffer = buffer.slice(header.headerLength);
        return parseBinaryVertices(dataBuffer, header, colorMode, fieldMap);
    }
    throw new Error(`Unsupported PLY format "${header.format}".`);
}

function parseAsciiVertices(dataText, schema, colorMode, extraFieldMap) {
    const { vertexCount, vertexProperties } = schema;
    const positions = new Float32Array(vertexCount * 3);
    const colors = new Uint8Array(vertexCount * 3);
    const rotations = new Float32Array(vertexCount * 4);
    const scales = new Float32Array(vertexCount * 3);
    const opacities = new Float32Array(vertexCount);
    const boundsMin = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const boundsMax = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    const colorFieldMap = buildColorFieldMap(colorMode);
    const shFieldMap = buildShFieldMap(colorMode);

    const lines = dataText.split(/\r?\n/);
    let consumed = 0;
    for (let i = 0; i < lines.length && consumed < vertexCount; i += 1) {
        const raw = lines[i].trim();
        if (!raw || raw.startsWith('comment')) {
            continue;
        }
        const tokens = raw.split(/\s+/);
        if (tokens.length < vertexProperties.length) {
            throw new Error(`Vertex row ${consumed} is missing properties.`);
        }

        let x = 0;
        let y = 0;
        let z = 0;
        const rgb = [0, 0, 0];
        let rgbHits = 0;
        const sh = [0, 0, 0];
        let shHits = 0;
        let opacity = 1.0;
        const quat = [1, 0, 0, 0];
        const sc = [0.01, 0.01, 0.01];

        for (let p = 0; p < vertexProperties.length; p += 1) {
            const prop = vertexProperties[p];
            const value = Number(tokens[p]);
            if (Number.isNaN(value)) {
                throw new Error(`Invalid value "${tokens[p]}" for property ${prop.name}.`);
            }
            if (prop.name === 'x') x = value;
            else if (prop.name === 'y') y = value;
            else if (prop.name === 'z') z = value;
            if (colorFieldMap && Object.prototype.hasOwnProperty.call(colorFieldMap, prop.name)) {
                const idx = colorFieldMap[prop.name];
                rgb[idx] = value;
                rgbHits += 1;
            }
            if (shFieldMap && Object.prototype.hasOwnProperty.call(shFieldMap, prop.name)) {
                const idx = shFieldMap[prop.name];
                sh[idx] = value;
                shHits += 1;
            }
            if (extraFieldMap.opacity && prop.name === extraFieldMap.opacity) {
                opacity = value;
            }
            const rotIdx = extraFieldMap.rotations.indexOf(prop.name);
            if (rotIdx !== -1) {
                quat[rotIdx] = value;
            }
            const scaleIdx = extraFieldMap.scales.indexOf(prop.name);
            if (scaleIdx !== -1) {
                sc[scaleIdx] = value;
            }
        }

        const base = consumed * 3;
        positions[base] = x;
        positions[base + 1] = y;
        positions[base + 2] = z;

        const color = finalizeColor(colorMode, rgb, rgbHits, sh, shHits);
        colors[base] = color[0];
        colors[base + 1] = color[1];
        colors[base + 2] = color[2];

        const rotBase = consumed * 4;
        rotations[rotBase] = quat[0];
        rotations[rotBase + 1] = quat[1];
        rotations[rotBase + 2] = quat[2];
        rotations[rotBase + 3] = quat[3];

        scales[base] = sc[0];
        scales[base + 1] = sc[1];
        scales[base + 2] = sc[2];
        opacities[consumed] = opacity;

        boundsMin[0] = Math.min(boundsMin[0], x);
        boundsMin[1] = Math.min(boundsMin[1], y);
        boundsMin[2] = Math.min(boundsMin[2], z);
        boundsMax[0] = Math.max(boundsMax[0], x);
        boundsMax[1] = Math.max(boundsMax[1], y);
        boundsMax[2] = Math.max(boundsMax[2], z);

        consumed += 1;
    }

    if (consumed !== vertexCount) {
        throw new Error(`PLY expected ${vertexCount} vertices, only parsed ${consumed}.`);
    }

    return {
        count: vertexCount,
        vertices: positions,
        colors,
        rotations,
        scales,
        opacities,
        bounds: { min: boundsMin, max: boundsMax },
    };
}

function parseBinaryVertices(dataBuffer, schema, colorMode, extraFieldMap) {
    const { vertexCount, vertexProperties } = schema;
    const positions = new Float32Array(vertexCount * 3);
    const colors = new Uint8Array(vertexCount * 3);
    const rotations = new Float32Array(vertexCount * 4);
    const scales = new Float32Array(vertexCount * 3);
    const opacities = new Float32Array(vertexCount);
    const boundsMin = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const boundsMax = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    const colorFieldMap = buildColorFieldMap(colorMode);
    const shFieldMap = buildShFieldMap(colorMode);

    const readers = vertexProperties.map((prop) => {
        const reader = BINARY_READERS[prop.type.toLowerCase()];
        if (!reader) {
            throw new Error(`Unsupported property type "${prop.type}" in binary PLY.`);
        }
        return reader;
    });

    const stride = readers.reduce((sum, entry) => sum + entry.size, 0);
    const requiredBytes = stride * vertexCount;
    if (dataBuffer.length < requiredBytes) {
        throw new Error(`Unexpected EOF while reading PLY vertices (need ${requiredBytes} bytes).`);
    }

    let offset = 0;
    for (let v = 0; v < vertexCount; v += 1) {
        let x = 0;
        let y = 0;
        let z = 0;
        const rgb = [0, 0, 0];
        let rgbHits = 0;
        const sh = [0, 0, 0];
        let shHits = 0;
        let opacity = 1.0;
        const quat = [1, 0, 0, 0];
        const sc = [0.01, 0.01, 0.01];

        for (let p = 0; p < vertexProperties.length; p += 1) {
            const prop = vertexProperties[p];
            const reader = readers[p];
            const value = reader.read(dataBuffer, offset);
            offset += reader.size;

            if (prop.name === 'x') x = value;
            else if (prop.name === 'y') y = value;
            else if (prop.name === 'z') z = value;

            if (colorFieldMap && Object.prototype.hasOwnProperty.call(colorFieldMap, prop.name)) {
                const idx = colorFieldMap[prop.name];
                rgb[idx] = value;
                rgbHits += 1;
            }
            if (shFieldMap && Object.prototype.hasOwnProperty.call(shFieldMap, prop.name)) {
                const idx = shFieldMap[prop.name];
                sh[idx] = value;
                shHits += 1;
            }
            if (extraFieldMap.opacity && prop.name === extraFieldMap.opacity) {
                opacity = value;
            }
            const rotIdx = extraFieldMap.rotations.indexOf(prop.name);
            if (rotIdx !== -1) {
                quat[rotIdx] = value;
            }
            const scaleIdx = extraFieldMap.scales.indexOf(prop.name);
            if (scaleIdx !== -1) {
                sc[scaleIdx] = value;
            }
        }

        const base = v * 3;
        positions[base] = x;
        positions[base + 1] = y;
        positions[base + 2] = z;

        const color = finalizeColor(colorMode, rgb, rgbHits, sh, shHits);
        colors[base] = color[0];
        colors[base + 1] = color[1];
        colors[base + 2] = color[2];

        const rotBase = v * 4;
        rotations[rotBase] = quat[0];
        rotations[rotBase + 1] = quat[1];
        rotations[rotBase + 2] = quat[2];
        rotations[rotBase + 3] = quat[3];

        scales[base] = sc[0];
        scales[base + 1] = sc[1];
        scales[base + 2] = sc[2];
        opacities[v] = opacity;

        boundsMin[0] = Math.min(boundsMin[0], x);
        boundsMin[1] = Math.min(boundsMin[1], y);
        boundsMin[2] = Math.min(boundsMin[2], z);
        boundsMax[0] = Math.max(boundsMax[0], x);
        boundsMax[1] = Math.max(boundsMax[1], y);
        boundsMax[2] = Math.max(boundsMax[2], z);
    }

    return {
        count: vertexCount,
        vertices: positions,
        colors,
        rotations,
        scales,
        opacities,
        bounds: { min: boundsMin, max: boundsMax },
    };
}

function finalizeColor(colorMode, rgb, rgbHits, sh, shHits) {
    if (colorMode.mode === 'rgb' && rgbHits === 3) {
        const needsScaling = rgb.some((v) => v > 1.0 + 1e-3);
        return rgb.map((v) => {
            const scaled = needsScaling ? v : v * 255.0;
            return clamp(Math.round(scaled), 0, 255);
        });
    }
    if (colorMode.mode === 'sh' && shHits === 3) {
        return sh.map((v) => clamp(Math.round(sigmoid(v) * 255.0), 0, 255));
    }
    return [220, 220, 220];
}

function computeDefaults(bounds) {
    const min = bounds.min;
    const max = bounds.max;
    const center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    const extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    const diag = Math.max(1e-6, Math.sqrt(dot(extent, extent)));
    const camera = [center[0], center[1], center[2] + diag * 1.6];
    return { center, camera };
}

function computeGaussianFootprint(opts) {
    const {
        right,
        up,
        focal,
        depth,
        scales,
        rotations,
        index,
        minSize,
        maxSize,
        pointScale,
    } = opts;

    const scaleIndex = index * 3;
    const rotIndex = index * 4;
    const sx = normalizeScale(scales[scaleIndex]);
    const sy = normalizeScale(scales[scaleIndex + 1]);
    const sz = normalizeScale(scales[scaleIndex + 2]);

    const qw = rotations[rotIndex];
    const qx = rotations[rotIndex + 1];
    const qy = rotations[rotIndex + 2];
    const qz = rotations[rotIndex + 3];
    const rotMatrix = quaternionToMatrix(qw, qx, qy, qz);
    const cols = [
        [rotMatrix[0], rotMatrix[3], rotMatrix[6]],
        [rotMatrix[1], rotMatrix[4], rotMatrix[7]],
        [rotMatrix[2], rotMatrix[5], rotMatrix[8]],
    ];
    const variances = [sx * sx, sy * sy, sz * sz];

    const dotComponents = cols.map((col) => ({
        u: dot(col, right),
        v: dot(col, up),
    }));

    let varU = 0;
    let varV = 0;
    let covUV = 0;
    for (let i = 0; i < 3; i += 1) {
        varU += variances[i] * dotComponents[i].u * dotComponents[i].u;
        varV += variances[i] * dotComponents[i].v * dotComponents[i].v;
        covUV += variances[i] * dotComponents[i].u * dotComponents[i].v;
    }

    const scaleFactor = (focal / Math.max(depth, 1e-3)) ** 2 * pointScale;
    varU = Math.max(varU * scaleFactor, 1e-8);
    varV = Math.max(varV * scaleFactor, 1e-8);
    covUV *= scaleFactor;

    const trace = varU + varV;
    const det = varU * varV - covUV * covUV;
    const discriminate = Math.sqrt(Math.max(0, trace * trace * 0.25 - det));
    const lambda1 = trace * 0.5 + discriminate;
    const lambda2 = trace * 0.5 - discriminate;

    let theta = 0;
    if (Math.abs(covUV) > 1e-6) {
        theta = 0.5 * Math.atan2(2 * covUV, varU - varV);
    } else if (varU < varV) {
        theta = Math.PI * 0.5;
    }

    const sigmaMajor = Math.sqrt(Math.max(lambda1, lambda2, 1e-9));
    const sigmaMinor = Math.sqrt(Math.max(Math.min(lambda1, lambda2), 1e-9));

    return {
        sigmaU: clamp(sigmaMajor, minSize, maxSize),
        sigmaV: clamp(sigmaMinor, minSize, maxSize),
        theta,
    };
}

function projectPoints(plyData, options) {
    const {
        camera,
        target,
        up,
        width,
        height,
        fov,
        maxPoints,
        pointScale,
        minPointSize,
        maxPointSize,
        margin,
    } = options;

    const { vertices, colors, rotations, scales, opacities, count } = plyData;
    const forward = normalize(subtract(target, camera));
    let upHint = up.slice();
    if (length(cross(forward, upHint)) < 1e-3) {
        upHint = [0, 0, 1];
    }
    const right = normalize(cross(forward, upHint));
    const trueUp = normalize(cross(right, forward));

    const focal = 0.5 * width / Math.tan((fov * Math.PI) / 360.0);

    const projected = [];
    let minDepth = Number.POSITIVE_INFINITY;
    let maxDepth = Number.NEGATIVE_INFINITY;

    for (let idx = 0; idx < count; idx += 1) {
        const base = idx * 3;
        const px = vertices[base];
        const py = vertices[base + 1];
        const pz = vertices[base + 2];

        const rel = [px - camera[0], py - camera[1], pz - camera[2]];
        const depth = dot(rel, forward);
        if (depth <= 1e-4) {
            continue;
        }
        const x = dot(rel, right);
        const y = dot(rel, trueUp);
        const u = focal * (x / depth) + width * 0.5;
        const v = -focal * (y / depth) + height * 0.5;

        if (u < -margin || u > width + margin || v < -margin || v > height + margin) {
            continue;
        }

        minDepth = Math.min(minDepth, depth);
        maxDepth = Math.max(maxDepth, depth);

        const footprint = computeGaussianFootprint({
            right,
            up: trueUp,
            focal,
            depth,
            scales,
            rotations,
            index: idx,
            minSize: minPointSize,
            maxSize: maxPointSize,
            pointScale,
        });

        projected.push({
            index: idx,
            u,
            v,
            depth,
            sigmaU: footprint.sigmaU,
            sigmaV: footprint.sigmaV,
            theta: footprint.theta,
            color: [colors[base], colors[base + 1], colors[base + 2]],
            opacity: normalizeOpacity(opacities[idx]),
        });
    }

    projected.sort((a, b) => a.depth - b.depth);

    const limited = enforcePointBudget(projected, maxPoints);
    return {
        points: limited,
        stats: { minDepth, maxDepth },
    };
}

function enforcePointBudget(points, maxPoints) {
    if (!maxPoints || maxPoints <= 0 || points.length <= maxPoints) {
        return points;
    }
    const limited = [];
    const step = (points.length - 1) / Math.max(1, (maxPoints - 1));
    for (let i = 0; i < maxPoints; i += 1) {
        const idx = Math.min(points.length - 1, Math.round(i * step));
        limited.push(points[idx]);
    }
    return limited;
}

function buildRenderPayload(points, stats, options) {
    const { minAlpha, maxAlpha } = options;
    const depthRange = Math.max(stats.maxDepth - stats.minDepth, 1e-5);
    return points.map((point) => {
        const normalizedDepth = clamp((point.depth - stats.minDepth) / depthRange, 0, 1);
        const baseAlpha = clamp(point.opacity, minAlpha, maxAlpha);
        const alpha = clamp(
            baseAlpha * (1.15 - 0.3 * normalizedDepth),
            minAlpha,
            maxAlpha,
        );
        const colorString = `rgba(${Math.round(point.color[0])},${Math.round(point.color[1])},${Math.round(point.color[2])},${alpha.toFixed(3)})`;
        return {
            u: point.u,
            v: point.v,
            sigmaU: point.sigmaU,
            sigmaV: point.sigmaV,
            theta: point.theta,
            color: colorString,
        };
    });
}

async function renderFrame(renderPayload, options) {
    const { width, height, background } = options;
    const canvas = new Canvas(width, height);
    const directCtx = canvas.getContext('2d');
    directCtx.save();
    directCtx.fillStyle = background;
    directCtx.fillRect(0, 0, width, height);
    directCtx.restore();
    const ctx = new JSContext();
    ctx.addAPI(new CanvasAPI(canvas));
    ctx.addAPI(new MathAPI());
    ctx.addAPI(new ColorsAPI());
    ctx.addAPI(new StringAPI());

    ctx.constant('renderPayload', renderPayload);
    ctx.constant('backgroundColor', background);

    const drawScript = `
        noPen();
        for (var i = 0; i < renderPayload.length; ++i) {
            var point = renderPayload[i];
            if (!point) continue;
            save();
            translate(point.u, point.v);
            rotate(point.theta);
            scale(point.sigmaU, point.sigmaV);
            fill(point.color);
            circle(0, 0, 1);
            restore();
        }
    `;

    ctx.evaluate(drawScript);
    const pngMaybe = canvas.toBuffer('png');
    const pngBuffer = pngMaybe instanceof Promise ? await pngMaybe : pngMaybe;
    ctx.destroy();
    return pngBuffer;
}

function spawnFfmpeg(videoPath, width, height, fps) {
    return spawn('ffmpeg', [
        '-y',
        '-f', 'image2pipe',
        '-vcodec', 'png',
        '-r', String(fps),
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        videoPath,
    ], { stdio: ['pipe', 'inherit', 'inherit'] });
}

async function renderVideoSequence(plyData, sequence, options) {
    const {
        videoPath,
        width,
        height,
        fov,
        background,
        maxPoints,
        pointScale,
        minPointSize,
        maxPointSize,
        minAlpha,
        maxAlpha,
        margin,
        fps,
    } = options;

    if (!sequence.length) {
        throw new Error('Camera sequence is empty.');
    }

    const ffmpeg = spawnFfmpeg(videoPath, width, height, fps);
    const ffmpegClosed = new Promise((resolve, reject) => {
        ffmpeg.on('error', reject);
        ffmpeg.on('close', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`ffmpeg exited with code ${code}`));
        });
    });

    try {
        const totalFrames = sequence.length;
        let renderedFrames = 0;
        for (const pose of sequence) {
            const projection = projectPoints(plyData, {
                camera: pose.camera,
                target: pose.target,
                up: pose.up,
                width,
                height,
                fov,
                maxPoints,
                pointScale,
                minPointSize,
                maxPointSize,
                margin,
            });
            if (!projection.points.length) {
                continue;
            }
            const renderPayload = buildRenderPayload(projection.points, projection.stats, {
                minAlpha,
                maxAlpha,
            });
            const png = await renderFrame(renderPayload, { width, height, background });
            ffmpeg.stdin.write(png);
            renderedFrames += 1;
            process.stdout.write(`\r[Render] Frame ${renderedFrames}/${totalFrames}`);
        }
        ffmpeg.stdin.end();
        await ffmpegClosed;
        if (sequence.length) {
            process.stdout.write('\n');
        }
    } catch (err) {
        ffmpeg.stdin.destroy();
        throw err;
    }
}

function loadCameraSequence(sequencePath, fallbackUp) {
    const raw = fs.readFileSync(sequencePath, 'utf8');
    const payload = JSON.parse(raw);
    if (!Array.isArray(payload) || !payload.length) {
        throw new Error('Camera sequence JSON must be a non-empty array.');
    }
    return payload.map((entry, idx) => {
        if (!entry.camera || !entry.target) {
            throw new Error(`Camera entry ${idx} missing camera/target vectors.`);
        }
        return {
            camera: parseVector(entry.camera, [0, 0, 0]),
            target: parseVector(entry.target, [0, 0, 0]),
            up: parseVector(entry.up || fallbackUp, fallbackUp),
        };
    });
}

function printHelp() {
    console.log(`Spark.js PLY renderer

Usage:
  Still: node scripts/spark_ply_renderer.js --ply input.ply --out still.png
  Video: node scripts/spark_ply_renderer.js --ply input.ply --video out.mp4 \
           --cameraSequence cameras.json --fps 12

Options:
  --ply <path>            Input PLY file (ASCII or binary_little_endian)
  --out <path>            Output PNG path for still rendering
  --video <path>          Output MP4 path (requires --cameraSequence)
  --camera <x,y,z>        Camera position override for stills
  --target <x,y,z>        Target point override for stills
  --up <x,y,z>            Up vector (default: 0,1,0)
  --cameraSequence <json> JSON camera path for video mode
  --width <px>            Output width (${DEFAULTS.width})
  --height <px>           Output height (${DEFAULTS.height})
  --fov <deg>             Field of view (${DEFAULTS.fov})
  --fps <value>           Video frames per second (${DEFAULTS.fps})
  --maxPoints <n>         Maximum splats (${DEFAULTS.maxPoints})
  --pointScale <value>    Screen footprint scale (${DEFAULTS.pointScale})
  --minPointSize <px>     Minimum projected radius (${DEFAULTS.minPointSize})
  --maxPointSize <px>     Maximum projected radius (${DEFAULTS.maxPointSize})
  --minAlpha <0-1>        Minimum opacity (${DEFAULTS.minAlpha})
  --maxAlpha <0-1>        Maximum opacity (${DEFAULTS.maxAlpha})
`);
}

async function main() {
    const argv = minimist(process.argv.slice(2), {
        string: ['ply', 'out', 'video', 'camera', 'target', 'up', 'background', 'cameraSequence'],
        boolean: ['help'],
        alias: { h: 'help' },
        default: DEFAULTS,
    });

    if (argv.help) {
        printHelp();
        process.exit(0);
    }

    if (!argv.ply) {
        console.error('Missing --ply=<path> argument.');
        printHelp();
        process.exit(1);
    }

    const plyPath = path.resolve(argv.ply);
    if (!fs.existsSync(plyPath)) {
        console.error(`PLY file not found: ${plyPath}`);
        process.exit(1);
    }

    const plyBuffer = fs.readFileSync(plyPath);
    const plyData = parsePly(plyBuffer);

    const width = coerceNumber(argv.width, DEFAULTS.width);
    const height = coerceNumber(argv.height, DEFAULTS.height);
    const fov = coerceNumber(argv.fov, DEFAULTS.fov);
    const maxPoints = coerceNumber(argv.maxPoints, DEFAULTS.maxPoints);
    const pointScale = coerceNumber(argv.pointScale, DEFAULTS.pointScale) / 1000.0;
    const minPointSize = coerceNumber(argv.minPointSize, DEFAULTS.minPointSize);
    const maxPointSize = coerceNumber(argv.maxPointSize, DEFAULTS.maxPointSize);
    const minAlpha = coerceNumber(argv.minAlpha, DEFAULTS.minAlpha);
    const maxAlpha = coerceNumber(argv.maxAlpha, DEFAULTS.maxAlpha);
    const background = argv.background || DEFAULTS.background;
    const margin = coerceNumber(argv.margin ?? DEFAULTS.margin, DEFAULTS.margin);

    const defaults = computeDefaults(plyData.bounds);

    const commonOptions = {
        width,
        height,
        fov,
        background,
        maxPoints,
        pointScale,
        minPointSize,
        maxPointSize,
        minAlpha,
        maxAlpha,
        margin,
    };

    if (argv.cameraSequence) {
        if (!argv.video) {
            throw new Error('Video mode requires --video <output.mp4>.');
        }
        const videoPath = path.resolve(argv.video);
        const fps = coerceNumber(argv.fps, DEFAULTS.fps);
        const sequence = loadCameraSequence(path.resolve(argv.cameraSequence), parseVector('0,1,0', [0, 1, 0]));
        await renderVideoSequence(plyData, sequence, {
            ...commonOptions,
            videoPath,
            fps,
        });
        console.log(`Video saved to ${videoPath}`);
        return;
    }

    if (!argv.out) {
        throw new Error('Still rendering requires --out <output.png>.');
    }

    const camera = parseVector(argv.camera, defaults.camera);
    const target = parseVector(argv.target, defaults.center);
    const up = parseVector(argv.up || '0,1,0', [0, 1, 0]);

    const projection = projectPoints(plyData, {
        camera,
        target,
        up,
        width,
        height,
        fov,
        maxPoints,
        pointScale,
        minPointSize,
        maxPointSize,
        margin,
    });

    if (!projection.points.length) {
        throw new Error('No vertices landed inside the camera frustum.');
    }

    const renderPayload = buildRenderPayload(projection.points, projection.stats, {
        minAlpha,
        maxAlpha,
    });
    const png = await renderFrame(renderPayload, { width, height, background });
    const outPath = path.resolve(argv.out);
    await fs.promises.mkdir(path.dirname(outPath), { recursive: true });
    await fs.promises.writeFile(outPath, png);
    console.log(`Rendered ${renderPayload.length} splats to ${outPath}`);
}

if (require.main === module) {
    main().catch((err) => {
        console.error(`Spark renderer failed: ${err.message}`);
        process.exit(1);
    });
}
