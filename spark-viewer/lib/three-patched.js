import * as THREE from './three.module.js';

// Re-export everything from original Three.js
export * from './three.module.js';

// Polyfill Matrix2
export class Matrix2 {
    constructor() {
        this.elements = [
            1, 0,
            0, 1
        ];
    }

    set(n11, n12, n21, n22) {
        const te = this.elements;
        te[0] = n11; te[2] = n12;
        te[1] = n21; te[3] = n22;
        return this;
    }

    identity() {
        this.set(1, 0, 0, 1);
        return this;
    }
    
    copy(m) {
        const te = this.elements;
        const me = m.elements;
        te[0] = me[0]; te[1] = me[1];
        te[2] = me[2]; te[3] = me[3];
        return this;
    }

    clone() {
        return new Matrix2().copy(this);
    }
}

// Polyfill BufferAttribute.prototype.addUpdateRange if missing
if (THREE.BufferAttribute && !THREE.BufferAttribute.prototype.addUpdateRange) {
    console.log("Polyfilling BufferAttribute.prototype.addUpdateRange");
    THREE.BufferAttribute.prototype.addUpdateRange = function ( start, count ) {
        const updateRange = this.updateRange;
        if ( updateRange.count === - 1 ) {
            updateRange.offset = start;
            updateRange.count = count;
        } else {
            const end = Math.max( updateRange.offset + updateRange.count, start + count );
            updateRange.offset = Math.min( updateRange.offset, start );
            updateRange.count = end - updateRange.offset;
        }
    };
}
