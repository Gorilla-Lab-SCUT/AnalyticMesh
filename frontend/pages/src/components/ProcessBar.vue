<template>
    <div>
        <div style="text-align:center;font-size:1rem;">{{title}}</div>
        <div class="processbar_box" :style="{height:height,width:width}">
            <div class="inner_box" ref="inner_box">
            </div>
            <div class="front_box">
            </div>
        </div>
        <div style="text-align:right;font-size:0.7rem;">
            {{text}}
        </div>
    </div>

</template>

<script>
    export default {
        name: "ProcessBar",
        props: {
            value: {
                default: 2
            },
            total: {
                default: 100
            },
            title: {
                default: 'Process Bar'
            },
            text: {
                default: '0/100'
            },
            height: {
                default: '10px'
            },
            width: {
                default: '50vw'
            }
        },
        watch: {
            value: function (val) {
                this.updateBox(val, this.total);
            },
            total: function (val) {
                this.updateBox(this.value, val);
            }
        },
        mounted() {
            this.updateBox(this.value, this.total);
        },
        methods: {
            updateBox: function (val, total) {
                let per;
                if (total <= val) {
                    per = 1;
                } else {
                    per = val / total;
                }
                this.$refs.inner_box.style = 'width:' + per * 100 + '%';
            }
        }
    }
</script>

<style scoped>
    .processbar_box {
        width: 100%;
        height: 100%;
        background-color: white;
        border-radius: 5px;
        display: flex;
        flex-direction: row;
        overflow: hidden;
    }

    .processbar_box .inner_box {
        background: dodgerblue;
        width: 100%;
        height: 100%;
        z-index: 2;
        border-radius: 5px;
        flex-shrink: 0;
    }

    .processbar_box .front_box {
        background: lightskyblue;
        width: 25px;
        height: 100%;
        animation: increasing linear infinite;
        animation-duration: 2s;
        border-radius: 5px;
        margin-left: -5px;
        flex-shrink: 1
    }

    @keyframes increasing {
        from {
            background: dodgerblue;
            transform: translateX(-20px);
        }
    }

</style>