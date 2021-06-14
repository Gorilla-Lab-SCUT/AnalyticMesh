<template>
    <div ref='main'>
        <div style="display: flex;flex-direction: row;" ref="root">
            <div ref="embedding_text" class="unSelect">
                <span :style="{fontSize:font_size,height:font_size,lineHeight:font_size}" class="embedding_text">{{text}}</span>
                <div v-if="text != ''" style="display:inline-block;width:10px">:</div>
            </div>
            <select ref='embedding' class="embedding_1" v-model="inner_value.method" @change="change_embedding">
                <option v-for="val,key in content" v-bind:key="key" :value="key">
                    {{key}}
                </option>
            </select>
        </div>
        <div class="info" v-show="info !== '' && hover" ref="info">
            <div>
                {{info}}
            </div>
        </div>
        <div v-show="inner_value.method != ''">
            <div class="embedding_box" ref="embedding_box">
                <div v-for="val,key in inner_value.args" v-bind:key="key"
                     style="padding:3px 6px;margin:2px;">
                    <v_Input :text="key" style="width:100%" :font_size="font_size" :active_color="active_color"
                             place_holder="please input number" v-model="inner_value.args[key]"
                             :rule="relus[key]" align="center" :info="infos[key]"></v_Input>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
    import v_Input from '@/components/Input.vue'

    export default {
        name: "TwoLayerEmbedding",
        components: {
            v_Input,
        },
        props: {
            font_size: {
                type: String,
                default: "1rem"
            },
            text: {
                type: String,
                default: ''
            },
            content: {
                type: Object,
                default: null
            },
            active_color: {
                type: String,
                default: 'blue'
            },
            value: {
                default: () => {
                    return {
                        method: '',
                        args: {}
                    }
                }
            },
            info: {
                type: String,
                default: ''
            }
        },
        data() {
            return {
                inner_value: {
                    method: '',
                    args: {}
                },
                VOXEL_SIZE: '',
                relus: {},
                infos: {},
                hover: false,
            }
        },
        watch: {
            value: function (val) {
                this.inner_value = val;
            },
            inner_value: function (val) {
                this.$emit('input', val);
            },
            deep: true
        },
        mounted() {
            this.inner_value = this.value;
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
        methods: {
            change_embedding: function () {
                this.inner_value.args = {};
                this.relus = {};
                for (let i in this.content[this.inner_value.method]) {
                    this.$set(this.inner_value.args, i, this.content[this.inner_value.method][i].default || '');
                    this.$set(this.relus, i, this.content[this.inner_value.method][i].relu || '.*');
                    this.$set(this.infos, i, this.content[this.inner_value.method][i].info || '')

                }
            },
        }
    }
</script>

<style scoped>
    .embedding_1 {
        flex: 1;
        text-align: center;
        text-align-last: center;
        border-radius: 5px;
        width: 100px;
        display: inline-block;
        outline: none;
        font-size: 1rem;
        background: transparent;
    }

    .embedding_text {
        display: inline-block;
    }

    .embedding_box {
        display: flex;
        flex-direction: column;
        border: 1px solid black;
        min-height: 50px;
        margin-top: 2px;
        border-radius: 5px;
        padding: 3px;
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