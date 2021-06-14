<template>
    <div>
        <div class="terminal_box">
            <div class="terminal" ref="terminal">
                <div v-for="(content,key) in contents" :key="key">
                    <div class="text">
                        <span style="color:red">{{content[0]}}</span>
                        <span style="margin-left:5px">{{content[1]}}</span>
                    </div>
                </div>
                <div class="cursor">_</div>
            </div>
        </div>
    </div>
</template>

<script>
    export default {
        name: "Terminal",
        data() {
            return {
                contents: []
            }
        },
        methods: {
            addLine: function (text) {
                Date.prototype.Format = function (fmt) { // author: meizz
                    var o = {
                        "M+": this.getMonth() + 1, // 月份
                        "d+": this.getDate(), // 日
                        "h+": this.getHours(), // 小时
                        "m+": this.getMinutes(), // 分
                        "s+": this.getSeconds(), // 秒
                        "q+": Math.floor((this.getMonth() + 3) / 3), // 季度
                        "S": this.getMilliseconds() // 毫秒
                    };
                    if (/(y+)/.test(fmt))
                        fmt = fmt.replace(RegExp.$1, (this.getFullYear() + "").substr(4 - RegExp.$1.length));
                    for (var k in o)
                        if (new RegExp("(" + k + ")").test(fmt)) fmt = fmt.replace(RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (("00" + o[k]).substr(("" + o[k]).length)));
                    return fmt;
                };
                this.contents.push([new Date().Format("yyyy/MM/dd hh:mm:ss"), text]);
                this.$nextTick(() => {
                    this.$refs.terminal.scrollTop = this.$refs.terminal.scrollHeight;
                })
            },
            clear: function () {
                this.contents = [];
            }
        }
    }
</script>

<style scoped>
    .terminal_box {
        border: 1px solid white;
        width: 100%;
        height: 100%;
        border-radius: 10px;
        overflow: hidden;
    }

    .terminal {
        background: black;
        width: calc(100% - 10px);
        height: calc(100% - 10px);
        overflow-y: auto;
        color: white;
        padding: 5px;
        text-align: left;
        font-size: 1.2rem;
    }

    .terminal::-webkit-scrollbar {
        /*滚动条整体样式*/
        width: 6px; /*高宽分别对应横竖滚动条的尺寸*/
        height: 6px;
    }

    .terminal::-webkit-scrollbar-thumb {
        /*滚动条里面小方块*/
        border-radius: 3px;
        box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.2);
        background: white;
    }

    .terminal::-webkit-scrollbar-track {
        /*滚动条里面轨道*/
        box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.2);
        background: black;
        border-radius: 3px;
    }

    @keyframes Blink {
        0%, 100% {
            opacity: 0;
        }
        50% {
            opacity: 1;
        }
    }

    .cursor {
        animation: Blink 1s infinite steps(1, start);
        display: inline-block;
    }

    .text {
        display: inline-block;
        word-break: break-all

    }
</style>