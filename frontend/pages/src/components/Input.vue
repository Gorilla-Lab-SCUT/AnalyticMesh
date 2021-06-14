<template>
    <div ref="root">
        <div style="display: flex;flex-direction: row;outline:none" class="input_div" tabindex="0" hidefocus="true"
             v-on:focus="onfocus" v-on:blur="onblur">
            <div v-show="text != ''">
                <div ref="input_text" class="unSelect">
                    <span :style="{fontSize:font_size,height:font_size,lineHeight:font_size}">{{text}}</span>
                    <div v-if="text != ''" style="display:inline-block;width:10px">:</div>
                </div>
            </div>
            <input class="input" ref='input' :style="{height:font_size,lineHeight:font_size,textAlign:align}"
                   v-on:keydown="keydown" v-on:focus="onfocus" v-on:blur="onblur" v-model="inner_value"
                   :placeholder="place_holder" required/>
        </div>
        <div class="info" v-show="info !== '' && hover" ref="info">
            <div>
                {{info}}
            </div>
        </div>
    </div>
</template>

<script>
    export default {
        name: "v_Input",

        props: {
            font_size: {
                type: String,
                default: "1.5rem"
            },
            text: {
                type: String,
                default: ""
            },
            active_color: {
                type: String,
                default: 'blue'
            },
            place_holder: {
                type: String,
                default: ''
            },
            value: {
                default: ''
            },
            rule: {
                type: String,
                default: '.*'
            },
            align: {
                type: String,
                default: 'center'
            },
            info: {
                type: String,
                default: ''
            }
        },
        data() {
            return {
                focus: false,
                inner_value: '',
                hover: false
            }
        },
        mounted() {
            this.inner_value = this.value;
            this.updateStyle(false);
            this.$refs.root.onmouseover = () => {
                this.hover = true;
            };
            this.$refs.root.onmouseout = () => {
                this.hover = false;
            };
            this.$refs.root.onmousemove = (ev) => {
                let e = ev || window.event;
                this.$refs.info.style.marginLeft = e.offsetX + 'px';
                this.$refs.info.style.marginTop = e.offsetY + 'px';
            };
        },
        watch: {
            focus: function (val) {
                this.updateStyle(val);
            },
            value: function (val) {
                this.inner_value = val;
            },
            inner_value: function (val) {
                let pattern = new RegExp(this.rule);
                let value = (val.toString().match(pattern) != null) ? val.toString().match(pattern)[0] : '';
                this.inner_value = value;
                this.$emit('input', value);
            },
        },
        methods: {
            updateStyle: function (val) {
                let style = ';font-size:' + this.font_size + ';height:' + this.font_size + ';line-height:' + this.font_size + ';text-align:' + this.align;
                if (val) {
                    this.$refs.input_text.style = 'color:' + this.active_color + style;
                    this.$refs.input.style = 'outline:none;border-bottom:1px solid' + this.active_color + ';color:' + this.active_color + style;
                } else {
                    this.$refs.input_text.style = style;
                    this.$refs.input.style = style;
                }
            },
            keydown: function (e) {
                if (e.keyCode == 13) {
                    e.preventDefault();
                }
                return true;
            },
            onfocus: function () {
                this.focus = true;
            },
            onblur: function () {
                this.focus = false;
            }
        },
    }
</script>

<style scoped>
    .input {
        border: 0;
        border-bottom: 1px solid black;
        overflow: hidden;
        word-wrap: unset;
        word-break: unset;
        display: inline-block;
        flex: 1;
        padding: 0 10px;
        background: transparent;
    }

    .info {
        position: absolute;
        z-index: 2;
        width: calc(100% - 40px);
        /*display: none;*/
    }

    .info * {
        background: gray;
        display: inline-block;
        padding: 5px;
        border-radius: 5px;
        color: white;
    }

</style>